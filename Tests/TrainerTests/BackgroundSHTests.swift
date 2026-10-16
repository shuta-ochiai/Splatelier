// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import MLXRandom
import Testing
@testable import Trainer

private let C0: Float = 0.28209479177387814

@Test func backgroundSHBandCount() {
    #expect(BackgroundSH.bandCount(degree: 0) == 1)
    #expect(BackgroundSH.bandCount(degree: 1) == 4)
    #expect(BackgroundSH.bandCount(degree: 2) == 9)
    #expect(BackgroundSH.bandCount(degree: 3) == 16)
}

@Test func backgroundSHRayDirectionsAreNormalizedAndForward() {
    // identity viewmat (R=I, t=0): world dir == normalized camera dir.
    let vm: [Float] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    let w = 8, h = 6
    let dirs = BackgroundSH.rayDirections(
        width: w, height: h, fx: 6, fy: 6, cx: 4, cy: 3, viewmatRowMajor: vm)
    eval(dirs)
    #expect(dirs.shape == [w * h, 3])
    let a = dirs.asArray(Float.self)
    for p in 0..<(w * h) {
        let x = a[p * 3], y = a[p * 3 + 1], z = a[p * 3 + 2]
        let n = (x * x + y * y + z * z).squareRoot()
        #expect(abs(n - 1) < 1e-4, "dir[\(p)] not normalized: \(n)")
        #expect(z > 0, "identity-cam ray must point forward (+z), got z=\(z) at \(p)")
    }
}

@Test func backgroundSHInitialCoeffsRenderGrey() {
    // The initial coefficients are DC only (0.5/C0), so every pixel is 0.5 regardless of degree
    // (uniform gray background at training start).
    let vm: [Float] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    let w = 5, h = 4
    for degree in 0...3 {
        let coeffs = BackgroundSH.initialCoeffs(degree: degree)
        #expect(coeffs.shape == [BackgroundSH.bandCount(degree: degree), 3])
        let bg = BackgroundSH.render(
            width: w, height: h, fx: 4, fy: 4, cx: 2.5, cy: 2,
            viewmatRowMajor: vm, coeffs: coeffs, degree: degree)
        eval(bg)
        #expect(bg.shape == [h, w, 3])
        let v = bg.reshaped([h * w * 3]).asArray(Float.self)
        var maxErr: Float = 0
        for x in v { maxErr = max(maxErr, abs(x - 0.5)) }
        #expect(maxErr < 1e-4, "degree \(degree): bg not uniform 0.5, maxErr=\(maxErr)")
    }
}

@Test func backgroundSHBasisDCColumnIsC0() {
    // The band-0 column of the basis is C0 for every direction.
    let dirs = MLXArray([
        Float(0), 0, 1,
        0, 1, 0,
        0.577, 0.577, 0.577,
    ], [3, 3])
    let b = BackgroundSH.basis(dirs: dirs, degree: 3)
    eval(b)
    #expect(b.shape == [3, 16])
    let col0 = b[0..., 0].asArray(Float.self)
    for c in col0 { #expect(abs(c - C0) < 1e-6, "DC basis must be C0, got \(c)") }
    // band-1 for direction (0,0,1) = (-C1·y, C1·z, -C1·x) = (0, C1, 0).
    let row0 = b[0, 0...].asArray(Float.self)
    let c1: Float = 0.4886025119029199
    #expect(abs(row0[1] - 0) < 1e-5)
    #expect(abs(row0[2] - c1) < 1e-5, "band Y10 for +z dir should be C1")
    #expect(abs(row0[3] - 0) < 1e-5)
}

// MARK: - Fused kernel (BackgroundSHKernels) vs pure MLX oracle

private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
    MLX.abs(a - b).max().item(Float.self)
}

// viewmat with a 90° rotation about y (row-major 4×4) so that R matters.
private let rotVM: [Float] = [
    0, 0, 1, 0,
    0, 1, 0, 0,
    -1, 0, 0, 0,
    0, 0, 0, 1,
]

/// The fused forward matches bg and composited against pure MLX (`BackgroundSH.render` + manual
/// compositing).
@Test func backgroundSHKernelForwardMatchesMLX() throws {
    let kernels = try BackgroundSHKernels()
    let w = 20, h = 15, deg = 3   // P=300 → 5 groups (the last is a partial group of 44)
    let fx: Float = 18, fy: Float = 18, cx: Float = 10, cy: Float = 7.5

    MLXRandom.seed(7)
    let coeffs = MLXRandom.normal([16, 3]) * Float(0.2)
    let render = MLXRandom.normal([h, w, 3]) * Float(0.1) + Float(0.5)
    let alpha = MLXRandom.uniform(low: 0.0, high: 1.0, [h, w])
    eval(coeffs, render, alpha)

    let (comp, bg) = kernels.forward(
        render: render, renderAlpha: alpha, coeffs: coeffs,
        width: w, height: h, degree: deg, fx: fx, fy: fy, cx: cx, cy: cy, viewmatRowMajor: rotVM)
    eval(comp, bg)

    let bgRef = BackgroundSH.render(
        width: w, height: h, fx: fx, fy: fy, cx: cx, cy: cy,
        viewmatRowMajor: rotVM, coeffs: coeffs, degree: deg)
    let omaRef = (1 - alpha).reshaped([h, w, 1])
    let compRef = render + omaRef * bgRef
    eval(bgRef, compRef)

    #expect(maxAbsDiff(bg, bgRef) < 1e-5, "bg: fused vs MLX = \(maxAbsDiff(bg, bgRef))")
    #expect(maxAbsDiff(comp, compRef) < 1e-5, "composited: \(maxAbsDiff(comp, compRef))")
    // Sanity: a non-trivial background actually comes out (not uniform gray).
    #expect(maxAbsDiff(bg, MLXArray(0.5 as Float)) > 1e-3, "bg is too uniform (test is not meaningful)")
}

/// The fused backward's coefficient gradient matches pure MLX (basisᵀ·dLdbg).
@Test func backgroundSHKernelBackwardMatchesMLX() throws {
    let kernels = try BackgroundSHKernels()
    let w = 20, h = 15, deg = 3
    let fx: Float = 18, fy: Float = 18, cx: Float = 10, cy: Float = 7.5

    MLXRandom.seed(13)
    let alpha = MLXRandom.uniform(low: 0.0, high: 1.0, [h, w])
    let vRender = MLXRandom.normal([h, w, 3]) * Float(0.02)
    eval(alpha, vRender)

    let dCoeffs = kernels.backwardCoeffs(
        vRender: vRender, renderAlpha: alpha,
        width: w, height: h, degree: deg, fx: fx, fy: fy, cx: cx, cy: cy, viewmatRowMajor: rotVM)
    eval(dCoeffs)

    // oracle: dL/dcoeffs = basisᵀ · (vRender·(1-alpha)).
    let dirs = BackgroundSH.rayDirections(
        width: w, height: h, fx: fx, fy: fy, cx: cx, cy: cy, viewmatRowMajor: rotVM)
    let basisM = BackgroundSH.basis(dirs: dirs, degree: deg)   // [P,16]
    let oma = (1 - alpha).reshaped([h * w, 1])
    let dLdbg = vRender.reshaped([h * w, 3]) * oma             // [P,3]
    // basisᵀ·dLdbg as a multiply and sum (a float32 matmul runs in TF32 on GPUs with matrix units).
    let dCoeffsRef = (basisM.expandedDimensions(axis: 2) * dLdbg.expandedDimensions(axis: 1))
        .sum(axis: 0)                                          // [16,3]
    eval(dCoeffsRef)

    #expect(dCoeffs.shape == [16, 3])
    #expect(maxAbsDiff(dCoeffs, dCoeffsRef) < 1e-5,
            "dCoeffs: fused vs MLX = \(maxAbsDiff(dCoeffs, dCoeffsRef))")
    #expect(MLX.abs(dCoeffsRef).max().item(Float.self) > 1e-6, "grad is nearly 0 (test is not meaningful)")
}

// MARK: - bg-SH bake export

private func makeTinyModel(n: Int) -> GaussianModel {
    var q = [Float](repeating: 0, count: n * 4)
    for i in 0..<n { q[i * 4] = 1 }
    return GaussianModel(
        means: MLXArray([Float](repeating: 0, count: n * 3), [n, 3]),
        quats: MLXArray(q, [n, 4]),
        logScales: MLXArray([Float](repeating: Foundation.log(0.01), count: n * 3), [n, 3]),
        logitOpacities: MLXArray([Float](repeating: 0, count: n), [n]),
        sh0: MLXArray.zeros([n, 3]),
        shN: MLXArray.zeros([n, GaussianModel.shRestBands, 3]))
}

/// Bake: baking a constant-color bg (DC only, coeff0=k/C0 → bg=k in every direction) gives every
/// env gaussian sh0 = (k-0.5)/C0 (the inverse of the viewer convention color=0.5+C0·f_dc). The
/// environment count also grows.
@Test func bakedBackgroundSphereConstantColor() {
    let n = 3, m = 256, deg = 3
    let model = makeTinyModel(n: n)
    let k: Float = 0.7
    var cf = [Float](repeating: 0, count: 16 * 3)
    cf[0] = k / C0; cf[1] = k / C0; cf[2] = k / C0   // DC only → bg=k in every direction
    let coeffs = MLXArray(cf, [16, 3])

    model.appendBakedBackgroundSphere(
        coeffs: coeffs, degree: deg,
        center: SIMD3<Float>(0, 0, 0), radius: 10, count: m, opacity: 0.9)
    eval(model.sh0)

    #expect(model.count == n + m)
    #expect(model.environmentCount == m)
    let expected = (k - 0.5) / C0
    let tail = model.sh0[n ..< (n + m), 0...].reshaped([m * 3]).asArray(Float.self)
    var maxErr: Float = 0
    for v in tail { maxErr = max(maxErr, abs(v - expected)) }
    #expect(maxErr < 1e-4, "baked sh0 != (k-0.5)/C0: maxErr=\(maxErr), expected=\(expected)")
}

/// Baking the gray initial coefficients gives sh0≈0 (color=0.5).
@Test func bakedBackgroundSphereGreyIsZeroSh0() {
    let model = makeTinyModel(n: 2)
    let coeffs = BackgroundSH.initialCoeffs(degree: 3)   // DC=0.5/C0 → bg=0.5
    model.appendBakedBackgroundSphere(
        coeffs: coeffs, degree: 3, center: SIMD3<Float>(0, 0, 0),
        radius: 5, count: 128, opacity: 0.9)
    eval(model.sh0)
    let tail = model.sh0[2 ..< model.count, 0...].reshaped([128 * 3]).asArray(Float.self)
    var maxAbs: Float = 0
    for v in tail { maxAbs = max(maxAbs, abs(v)) }
    #expect(maxAbs < 1e-4, "grey bg should bake sh0≈0, got maxAbs=\(maxAbs)")
}
