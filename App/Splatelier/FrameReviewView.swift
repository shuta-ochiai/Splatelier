// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Post-capture frame review. Shows all frames in a thumbnail grid; dropped frames are dimmed and
// marked with a reason badge. The verdict is the same CaptureFrameQualityFilter.review used by
// training.

import SwiftUI
import MLX
import Capture

struct FrameReviewView: View {
    let scene: CapturedScene
    /// Whether the capture setting's quality filter is enabled. If false, every frame is an image
    /// used for training.
    let qualityFilterEnabled: Bool

    @State private var enlarged: EnlargedItem?
    @Environment(\.dismiss) private var dismiss

    struct EnlargedItem: Identifiable {
        let index: Int
        var id: Int { index }
    }

    /// Per-frame keep/drop verdict (same as training). All frames are evaluated on the GPU, so it
    /// is computed only once at display time (on main, because CapturedFrame is not Sendable).
    @State private var reviews: [CaptureFrameQualityFilter.FrameReview] = []

    private static func computeReviews(
        scene: CapturedScene, qualityFilterEnabled: Bool
    ) -> [CaptureFrameQualityFilter.FrameReview] {
        if qualityFilterEnabled {
            return CaptureFrameQualityFilter.review(scene: scene)
        }
        return scene.frames.indices.map {
            .init(index: $0, kept: true, reasons: [], sharpness: 0)
        }
    }

    private var keptCount: Int { reviews.filter(\.kept).count }
    private var droppedCount: Int { reviews.count - keptCount }
    /// Number of frames that were flagged but kept to secure the minimum count (yellow badge).
    private var rescuedCount: Int { reviews.filter { $0.kept && !$0.reasons.isEmpty }.count }

    private let columns = [GridItem(.adaptive(minimum: 100), spacing: 6)]

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                if reviews.isEmpty && !scene.frames.isEmpty {
                    ProgressView("Checking…").padding(.vertical, 8)
                }
                VStack(spacing: 2) {
                    Text("\(reviews.count) frames / \(keptCount) used for training · \(droppedCount) excluded")
                    Text("Red = excluded (with reason) · Yellow = rescued\(rescuedCount > 0 ? " (\(rescuedCount))" : "") (kept to meet the minimum frame count)")
                        .foregroundStyle(.secondary)
                }
                .font(.footnote)
                .multilineTextAlignment(.center)
                .padding(.vertical, 8)

                ScrollView {
                    LazyVGrid(columns: columns, spacing: 6) {
                        ForEach(reviews, id: \.index) { r in
                            FrameThumbnail(
                                frame: scene.frames[r.index],
                                index: r.index,
                                dropped: !r.kept,
                                reasons: r.reasons,
                                blurScore: r.blurScore,
                                onTap: { enlarged = EnlargedItem(index: r.index) }
                            )
                        }
                    }
                    .padding(.horizontal, 6)
                }
            }
            .navigationTitle("Review frames")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Close") { dismiss() }
                }
            }
            .fullScreenCover(item: $enlarged) { item in
                EnlargedPagerView(scene: scene, reviews: reviews, startFrameIndex: item.index)
            }
            .task {
                if reviews.isEmpty {
                    reviews = Self.computeReviews(scene: scene, qualityFilterEnabled: qualityFilterEnabled)
                }
            }
        }
    }
}

/// Full-screen pager opened by tap. Swipe through all frames.
private struct EnlargedPagerView: View {
    let scene: CapturedScene
    let reviews: [CaptureFrameQualityFilter.FrameReview]
    let startFrameIndex: Int
    @Environment(\.dismiss) private var dismiss
    @State private var selection: Int   // Position within the reviews array

    init(scene: CapturedScene,
         reviews: [CaptureFrameQualityFilter.FrameReview],
         startFrameIndex: Int) {
        self.scene = scene
        self.reviews = reviews
        self.startFrameIndex = startFrameIndex
        _selection = State(initialValue:
            reviews.firstIndex { $0.index == startFrameIndex } ?? 0)
    }

    var body: some View {
        ZStack(alignment: .top) {
            Color.black.ignoresSafeArea()
            TabView(selection: $selection) {
                ForEach(reviews.indices, id: \.self) { i in
                    EnlargedFramePage(frame: scene.frames[reviews[i].index], review: reviews[i])
                        .tag(i)
                }
            }
            .tabViewStyle(.page(indexDisplayMode: .never))
            .ignoresSafeArea()

            HStack {
                Text("frame \(currentReview.index)" + (currentReview.kept ? "" : " (excluded)"))
                    .font(.headline)
                    .foregroundStyle(currentReview.kept ? .white : .red)
                Text("\(selection + 1)/\(reviews.count)")
                    .font(.subheadline.monospacedDigit())
                    .foregroundStyle(.white.opacity(0.7))
                Spacer()
                Button("Close") { dismiss() }.foregroundStyle(.white)
            }
            .padding()
            .background(.black.opacity(0.4))
        }
    }

    private var currentReview: CaptureFrameQualityFilter.FrameReview {
        reviews.indices.contains(selection) ? reviews[selection] : reviews[0]
    }
}

/// Turn the quality filter's reason strings into display labels.
private func blurReasonLabels(_ reasons: [String]) -> String {
    reasons.map {
        switch $0 {
        case "blur": return "Blur"
        case "localblur": return "Local blur"
        case "exposure": return "Exposure"
        default: return $0
        }
    }.joined(separator: "+")
}

/// Verdict state of a frame. Even if kept, having reasons means "Rescued" (kept to secure the
/// minimum count).
private enum FrameVerdict {
    case clean
    case rescued([String])
    case dropped([String])

    init(kept: Bool, reasons: [String]) {
        if !kept { self = .dropped(reasons) }
        else if reasons.isEmpty { self = .clean }
        else { self = .rescued(reasons) }
    }
}

/// One page in the pager. Shows the verdict and blur at the bottom.
private struct EnlargedFramePage: View {
    let frame: CapturedFrame
    let review: CaptureFrameQualityFilter.FrameReview
    @State private var image: UIImage?

    var body: some View {
        ZStack {
            Color.black
            if let image {
                Image(uiImage: image).resizable().scaledToFit()
            } else {
                ProgressView().tint(.white)
            }
            VStack {
                Spacer()
                HStack(spacing: 8) {
                    switch FrameVerdict(kept: review.kept, reasons: review.reasons) {
                    case .dropped(let r):
                        Text("Excluded: \(blurReasonLabels(r))")
                            .font(.subheadline.weight(.semibold))
                            .padding(.horizontal, 10).padding(.vertical, 5)
                            .background(.red, in: Capsule())
                            .foregroundStyle(.white)
                    case .rescued(let r):
                        Text("Rescued (flagged \(blurReasonLabels(r)), kept to meet the minimum frame count)")
                            .font(.subheadline.weight(.semibold))
                            .padding(.horizontal, 10).padding(.vertical, 5)
                            .background(.yellow, in: Capsule())
                            .foregroundStyle(.black)
                    case .clean:
                        EmptyView()
                    }
                    Text(String(format: "blur %.3f", review.blurScore))
                        .font(.subheadline.monospacedDigit())
                        .padding(.horizontal, 10).padding(.vertical, 5)
                        .background(review.blurScore > 0.5 ? .orange.opacity(0.9) : .black.opacity(0.55),
                                    in: Capsule())
                        .foregroundStyle(.white)
                }
                .padding(.bottom, 32)
            }
        }
        // Separately from the thumbnail (~200px), generate a high-resolution image only for the
        // page being shown.
        .task(id: review.index) {
            if image == nil { image = FrameThumbnail.makeThumbnail(frame.image, maxDim: 1200) }
        }
    }
}

/// Thumbnail of one frame. Dropped frames are dimmed + red border + reason badge.
private struct FrameThumbnail: View {
    let frame: CapturedFrame
    let index: Int
    let dropped: Bool
    /// Verdict reasons ("blur" / "localblur" / "exposure"). Non-empty even when kept means "Rescued"
    /// (kept anyway).
    var reasons: [String] = []
    /// Absolute blur score [0,1] (0 = sharp). Shown as a number for threshold calibration.
    var blurScore: Float = 0
    var onTap: () -> Void = {}

    @State private var image: UIImage?

    private var verdict: FrameVerdict { FrameVerdict(kept: !dropped, reasons: reasons) }

    var body: some View {
        ZStack(alignment: .topLeading) {
            Group {
                if let image {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFill()
                } else {
                    Rectangle().fill(.quaternary)
                }
            }
            .frame(width: 100, height: 75)
            .clipped()
            .overlay(alignment: .bottomTrailing) {
                Text("\(index)")
                    .font(.caption2.monospacedDigit())
                    .padding(2)
                    .background(.black.opacity(0.5), in: RoundedRectangle(cornerRadius: 3))
                    .foregroundStyle(.white)
                    .padding(2)
            }
            .overlay {
                if dropped {
                    Rectangle()
                        .fill(.black.opacity(0.45))
                        .overlay(RoundedRectangle(cornerRadius: 4).stroke(.red, lineWidth: 2))
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 4))

            VStack(alignment: .leading, spacing: 2) {
                switch verdict {
                case .dropped(let r):
                    Text("Excluded: \(blurReasonLabels(r))")
                        .font(.caption2.weight(.semibold))
                        .padding(.horizontal, 4).padding(.vertical, 1)
                        .background(.red, in: Capsule())
                        .foregroundStyle(.white)
                case .rescued:
                    Text("Rescued")
                        .font(.caption2.weight(.semibold))
                        .padding(.horizontal, 4).padding(.vertical, 1)
                        .background(.yellow, in: Capsule())
                        .foregroundStyle(.black)
                case .clean:
                    EmptyView()
                }
                // Show 3 digits so values near the threshold are not blurred together by rounding.
                Text(String(format: "b%.3f", blurScore))
                    .font(.caption2.monospacedDigit())
                    .padding(.horizontal, 3).padding(.vertical, 1)
                    .background(blurScore > 0.5 ? .orange.opacity(0.85) : .black.opacity(0.5),
                                in: RoundedRectangle(cornerRadius: 3))
                    .foregroundStyle(.white)
            }
            .padding(3)
        }
        .task(id: index) {
            if image == nil { image = Self.makeThumbnail(frame.image) }
        }
        .contentShape(Rectangle())
        .onTapGesture { onTap() }
    }

    /// Stride-decimate a uint8 [H,W,3] image to ~maxDim and convert to UIImage
    /// (holding 100+ frames at full res uses too much memory).
    static func makeThumbnail(_ image: MLXArray, maxDim: Int = 200) -> UIImage? {
        guard image.shape.count == 3, image.shape[2] == 3 else { return nil }
        let h = image.shape[0], w = image.shape[1]
        guard h > 0, w > 0 else { return nil }
        let s = Swift.max(1, Swift.max(h, w) / maxDim)
        let tw = Swift.max(1, w / s), th = Swift.max(1, h / s)
        // Decimate to every s-th pixel on the GPU before reading back (do not copy full res to the
        // host). sampled is [th, tw, 3].
        let src = image[.stride(from: 0, to: th * s, by: s), .stride(from: 0, to: tw * s, by: s), 0...]
            .reshaped([th * tw * 3]).asArray(UInt8.self)
        // The capture buffer appears rotated 90° counterclockwise on screen, so rotate clockwise
        // while decimating. out[i][j] = sampled[th-1-j][i]; output is width=th, height=tw.
        let outW = th, outH = tw
        var dst = [UInt8](repeating: 0, count: outW * outH * 3)
        for i in 0..<outH {              // Output row (0..<tw)
            for j in 0..<outW {          // Output column (0..<th)
                let si = ((th - 1 - j) * tw + i) * 3
                let di = (i * outW + j) * 3
                dst[di] = src[si]; dst[di + 1] = src[si + 1]; dst[di + 2] = src[si + 2]
            }
        }
        let data = Data(dst)
        guard let provider = CGDataProvider(data: data as CFData) else { return nil }
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let cg = CGImage(
            width: outW, height: outH,
            bitsPerComponent: 8, bitsPerPixel: 24, bytesPerRow: outW * 3,
            space: colorSpace,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
            provider: provider, decode: nil, shouldInterpolate: true,
            intent: .defaultIntent
        ) else { return nil }
        return UIImage(cgImage: cg)
    }
}
