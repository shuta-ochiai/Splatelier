// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Learnable directional SH background (splatfacto-W style). The background is SH over the ray
// direction:
//   bg(pixel) = Σ_b coeffs[b] · Y_b(dir(pixel))   (dir = world-space ray)
//   final     = renderColor(foreground, black bg) + (1 - alpha) · bg
// Cost is O(pixels) and adds no foreground rasterization. Per-view bases are not cached (huge at
// full res, and keeping one for every view under random-view training OOMs). The SH basis matches
// `sh_eval_deg3` in `gs_math.slang` (extended to degree 4) but is the raw value, without the +0.5 /
// clamp.

import Foundation
import MLX
import MLXFast

public enum BackgroundSH {

    // Same normalization constants as gs_math.slang sh_eval_deg3.
    private static let C0 = GaussianModel.shC0
    private static let C1: Float = 0.4886025119029199
    private static let C2_0: Float = 1.0925484305920792
    private static let C2_1: Float = -1.0925484305920792
    private static let C2_2: Float = 0.31539156525252005
    private static let C2_3: Float = -1.0925484305920792
    private static let C2_4: Float = 0.5462742152960396
    private static let C3_0: Float = -0.5900435899266435
    private static let C3_1: Float = 2.890611442640554
    private static let C3_2: Float = -0.4570457994644658
    private static let C3_3: Float = 0.3731763325901154
    private static let C3_4: Float = -0.4570457994644658
    private static let C3_5: Float = 1.445305721320277
    private static let C3_6: Float = -0.5900435899266435
    // Degree 4 (9 terms). Standard 3DGS SH_C4. Must match background_sh.slang exactly.
    private static let C4_0: Float = 2.5033429417967046
    private static let C4_1: Float = -1.7701307697799304
    private static let C4_2: Float = 0.9461746957575601
    private static let C4_3: Float = -0.6690465435572892
    private static let C4_4: Float = 0.10578554691520431
    private static let C4_5: Float = -0.6690465435572892
    private static let C4_6: Float = 0.47308734787878004
    private static let C4_7: Float = -1.7701307697799304
    private static let C4_8: Float = 0.6258357354491761

    /// Coefficient count (d+1)² for degree d. The bg-SH kernel supports up to degree 4 (25).
    public static func bandCount(degree: Int) -> Int {
        let d = Swift.max(0, Swift.min(4, degree))
        return (d + 1) * (d + 1)
    }

    /// Per-pixel world-space ray directions [P, 3] (normalized, P = width·height, row-major).
    /// cam_dir = ((px+0.5-cx)/fx, (py+0.5-cy)/fy, 1) is mapped back by world_dir = Rᵀ·cam_dir (+z
    /// forward, same as projection).
    public static func rayDirections(
        width: Int, height: Int,
        fx: Float, fy: Float, cx: Float, cy: Float,
        viewmatRowMajor vm: [Float]
    ) -> MLXArray {
        precondition(vm.count == 16, "viewmat must be row-major 4×4")
        precondition(width > 0 && height > 0, "image dims must be positive")
        let xs = (MLXArray(stride(from: 0, to: width, by: 1).map { Float($0) + 0.5 }, [width]) - cx) / fx   // [W]
        let ys = (MLXArray(stride(from: 0, to: height, by: 1).map { Float($0) + 0.5 }, [height]) - cy) / fy // [H]
        let zerosHW = MLXArray.zeros([height, width])
        let xGrid = xs.reshaped([1, width]) + zerosHW     // [H,W]
        let yGrid = ys.reshaped([height, 1]) + zerosHW    // [H,W]
        let zGrid = MLXArray.ones([height, width])
        let camDir = MLX.stacked([xGrid, yGrid, zGrid], axis: 2).reshaped([height * width, 3])   // [P,3] unnormalized
        // world→cam rotation R (row-major). With row vectors, world_dir = cam_dir · R (a multiply
        // and sum, see `colors`).
        let r = MLXArray(
            [vm[0], vm[1], vm[2],
             vm[4], vm[5], vm[6],
             vm[8], vm[9], vm[10]], [3, 3]
        )
        let worldDir = (camDir.expandedDimensions(axis: 2) * r.expandedDimensions(axis: 0))
            .sum(axis: 1)                                                      // [P,3]
        let sq = (worldDir * worldDir).sum(axis: 1, keepDims: true)            // [P,1]
        let norm = 1.0 / MLX.sqrt(MLX.maximum(sq, MLXArray(1e-20 as Float)))
        return worldDir * norm                                                // [P,3] normalized
    }

    /// SH basis [P, bands] for unit directions [P, 3] (column order as in `sh_eval_deg3`). bg =
    /// basis · coeffs (`colors`).
    public static func basis(dirs: MLXArray, degree: Int) -> MLXArray {
        precondition(dirs.shape.count == 2 && dirs.shape[1] == 3, "dirs must be [P,3]")
        let d = Swift.max(0, Swift.min(4, degree))
        let p = dirs.shape[0]
        let x = dirs[0..., 0]   // [P]
        let y = dirs[0..., 1]
        let z = dirs[0..., 2]
        var cols: [MLXArray] = [MLXArray([Float](repeating: C0, count: p), [p])]   // band 0
        if d >= 1 {
            cols.append(-C1 * y)
            cols.append(C1 * z)
            cols.append(-C1 * x)
        }
        if d >= 2 {
            let xx = x * x, yy = y * y, zz = z * z
            cols.append(C2_0 * (x * y))
            cols.append(C2_1 * (y * z))
            cols.append(C2_2 * (2.0 * zz - xx - yy))
            cols.append(C2_3 * (x * z))
            cols.append(C2_4 * (xx - yy))
        }
        if d >= 3 {
            let xx = x * x, yy = y * y, zz = z * z
            cols.append(C3_0 * y * (3.0 * xx - yy))
            cols.append(C3_1 * (x * y) * z)
            cols.append(C3_2 * y * (4.0 * zz - xx - yy))
            cols.append(C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy))
            cols.append(C3_4 * x * (4.0 * zz - xx - yy))
            cols.append(C3_5 * z * (xx - yy))
            cols.append(C3_6 * x * (xx - 3.0 * yy))
        }
        if d >= 4 {
            let xx = x * x, yy = y * y, zz = z * z
            cols.append(C4_0 * (x * y) * (xx - yy))
            cols.append(C4_1 * (y * z) * (3.0 * xx - yy))
            cols.append(C4_2 * (x * y) * (7.0 * zz - 1.0))
            cols.append(C4_3 * (y * z) * (7.0 * zz - 3.0))
            cols.append(C4_4 * (35.0 * zz * zz - 30.0 * zz + 3.0))
            cols.append(C4_5 * (x * z) * (7.0 * zz - 3.0))
            cols.append(C4_6 * (xx - yy) * (7.0 * zz - 1.0))
            cols.append(C4_7 * (x * z) * (xx - 3.0 * yy))
            cols.append(C4_8 * (xx * (xx - 3.0 * yy) - yy * (3.0 * xx - yy)))
        }
        return MLX.stacked(cols, axis: 1)   // [P, bands]
    }

    /// Initial coefficients [bands, 3]. DC gives a mid-grey bg = C0·coeffs[0] = 0.5; higher orders
    /// are 0.
    public static func initialCoeffs(degree: Int) -> MLXArray {
        let bands = bandCount(degree: degree)
        var flat = [Float](repeating: 0, count: bands * 3)
        let dc: Float = 0.5 / C0
        flat[0] = dc; flat[1] = dc; flat[2] = dc   // band 0, RGB
        return MLXArray(flat, [bands, 3])
    }

    /// Raw bg colors [P, 3] = basis(dirs) · coeffs for unit directions [P, 3]. A multiply and sum
    /// over the bands rather than `matmul`: MLX runs a float32 matmul with more than one row and
    /// column in TF32 (about 1e-3 relative) on GPUs with matrix units (M5, A19).
    public static func colors(dirs: MLXArray, coeffs: MLXArray, degree: Int) -> MLXArray {
        let b = basis(dirs: dirs, degree: degree)   // [P, bands]
        return (b.expandedDimensions(axis: 2) * coeffs.expandedDimensions(axis: 0)).sum(axis: 1)
    }

    /// bg image [H, W, 3] (basis recomputed each call). `coeffs` is [bands, 3].
    public static func render(
        width: Int, height: Int,
        fx: Float, fy: Float, cx: Float, cy: Float,
        viewmatRowMajor vm: [Float], coeffs: MLXArray, degree: Int
    ) -> MLXArray {
        let dirs = rayDirections(width: width, height: height,
                                 fx: fx, fy: fy, cx: cx, cy: cy, viewmatRowMajor: vm)
        return colors(dirs: dirs, coeffs: coeffs, degree: degree).reshaped([height, width, 3])
    }
}

/// Fused bg-SH kernel. Forward does ray, basis, dot product and compositing per pixel in registers
/// (no basis materialization); backward gets coefficient gradients via threadgroup reduction. The
/// math matches `BackgroundSH`, which serves as the verification oracle.
public final class BackgroundSHKernels {
    private let fwd: MLXFast.MLXFastKernel
    private let bwd: MLXFast.MLXFastKernel

    public init() throws {
        self.fwd = try SlangCompiledKernel.load(name: "background_sh_fwd")
        self.bwd = try SlangCompiledKernel.load(name: "background_sh_bwd")
    }

    /// Returns `composited = render + (1-alpha)·bg` and `bg` (both [H, W, 3]).
    /// render / renderAlpha are flattened internally. `coeffs` is [(degree+1)², 3].
    public func forward(
        render: MLXArray, renderAlpha: MLXArray, coeffs: MLXArray,
        width w: Int, height h: Int, degree: Int,
        fx: Float, fy: Float, cx: Float, cy: Float, viewmatRowMajor vm: [Float]
    ) -> (composited: MLXArray, bg: MLXArray) {
        let p = w * h
        let out = fwd(
            [render.reshaped([p * 3]), renderAlpha.reshaped([p]), coeffs.reshaped([coeffs.size]),
             MLXArray(vm, [16]), MLXArray([fx, fy, cx, cy], [4]),
             MLXArray([Int32(w), Int32(h), Int32(degree)], [3])],
            grid: (((p + 63) / 64) * 64, 1, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [[p * 3], [p * 3]],
            outputDTypes: [.float32, .float32]
        )
        return (out[0].reshaped([h, w, 3]), out[1].reshaped([h, w, 3]))
    }

    /// Returns dL/dcoeffs [bands, 3] (`vRender` = dL/dcomposited). The kernel emits per-threadgroup
    /// partial sums [numGroups, 75], summed in MLX. bg is linear in the coefficients, so coeffs is
    /// not an input.
    public func backwardCoeffs(
        vRender: MLXArray, renderAlpha: MLXArray,
        width w: Int, height h: Int, degree: Int,
        fx: Float, fy: Float, cx: Float, cy: Float, viewmatRowMajor vm: [Float]
    ) -> MLXArray {
        let p = w * h
        let numGroups = (p + 63) / 64
        let bands = (degree + 1) * (degree + 1)
        let out = bwd(
            [vRender.reshaped([p * 3]), renderAlpha.reshaped([p]),
             MLXArray(vm, [16]), MLXArray([fx, fy, cx, cy], [4]),
             MLXArray([Int32(w), Int32(h), Int32(degree)], [3])],
            grid: (numGroups * 64, 1, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [[numGroups * 75]],
            outputDTypes: [.float32]
        )
        let summed = out[0].reshaped([numGroups, 75]).sum(axis: 0)   // [75] = 25 degree-4 coefficients × 3
        return summed[0 ..< (bands * 3)].reshaped([bands, 3])
    }
}
