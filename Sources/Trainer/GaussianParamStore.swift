// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import MLX

/// Parameter group identifiers. Densification distinguishes groups that need correction (opacity,
/// scale) from those that are only copied.
public enum GaussianParamKind: CaseIterable {
    case means, quats, logScales, logitOpacities, sh0, shN
}

/// Training parameters paired with optimizer moments. Densification rewrites both together
/// (equivalent to gsplat's params + optimizers and `_update_param_with_optimizer`).
public protocol GaussianParamStore: AnyObject {
    var model: GaussianModel { get }
    var gaussianCount: Int { get }

    /// Rewrites all parameters together with Adam's m / v (momentFn is called for each of m and v).
    /// Row counts of all parameters and moment shapes must match the new parameters (else a shape
    /// mismatch on the next step). Adam's stepCount is preserved.
    func updateParamsWithOptimizer(
        paramFn: (GaussianParamKind, MLXArray) -> MLXArray,
        momentFn: (GaussianParamKind, MLXArray) -> MLXArray
    )
}
