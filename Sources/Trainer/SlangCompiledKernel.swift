// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import MLXFast

/// Reads the `.metal` + `.refl.json` emitted by slangc and wraps them in `MLXFast.MLXFastKernel`.
/// MLX takes only the body and generates the signature itself, so this parses the signature, splits
/// buffers into inputs and outputs (reflection readWrite = output), strips `KernelContext_*`, and
/// prepends aliases for semantic arguments. Inputs are bound positionally in buffer-index order
/// (callers must pass them in Slang declaration order).
enum SlangCompiledKernel {

    enum Error: Swift.Error, CustomStringConvertible {
        case resourceNotFound(String)
        case parseFailed(String)
        case decodeFailed(String)
        case unsupportedBindingLayout(String)

        var description: String {
            switch self {
            case .resourceNotFound(let s): return "Slang resource not found: \(s)"
            case .parseFailed(let s): return "Slang .metal parse failed: \(s)"
            case .decodeFailed(let s): return "Slang reflection decode failed: \(s)"
            case .unsupportedBindingLayout(let s): return "Slang binding layout not supported by MLX: \(s)"
            }
        }
    }

    /// Loads a bundled kernel. Failure is an unrecoverable build defect, so it is a fatalError.
    static func required(_ name: String, atomicOutputs: Bool = false) -> MLXFast.MLXFastKernel {
        do { return try load(name: name, atomicOutputs: atomicOutputs) }
        catch { fatalError("Failed to load \(name) Slang kernel: \(error)") }
    }

    /// Builds a kernel from the Slang artifacts for `name`.
    /// - Parameter atomicOutputs: Declare outputs as `device atomic<float>*`. Needed by kernels
    ///   that
    ///   `InterlockedAdd` into the same cell with high parallelism (BA Schur etc.); with false,
    ///   values drift
    ///   under contention.
    static func load(name: String, bundle: Bundle = .module, atomicOutputs: Bool = false) throws -> MLXFast.MLXFastKernel {
        let metalSource = try loadResource(name: name, extension: "metal", bundle: bundle)
        let reflJSON = try loadResource(name: name, extension: "refl.json", bundle: bundle)

        let reflection = try parseReflection(reflJSON)
        let parsed = try parseMetal(metalSource)

        // Use reflection to classify each buffer index as input (read-only) or output (readWrite).
        var accessByIndex: [Int: Bool] = [:]
        for p in reflection.parameters {
            accessByIndex[p.binding.index] = (p.type.access == "readWrite")
        }

        // MLX renumbers indices from 0 in inputNames + outputNames order, but the body refers to
        // mangled names, so this is safe.
        let sortedBufferParams = parsed.bufferParams.sorted { $0.bufferIndex < $1.bufferIndex }
        var inputNames: [String] = []
        var outputNames: [String] = []
        for bp in sortedBufferParams {
            let isOutput = accessByIndex[bp.bufferIndex] ?? false
            if isOutput {
                outputNames.append(bp.name)
            } else {
                inputNames.append(bp.name)
            }
        }

        let bodyClean = stripKernelContext(parsed.body)
        // MLX auto-declares in the signature any Metal builtin attribute names that appear in the
        // source. Slang receives builtins under its own names, so prepend `<slangName> =
        // <metalAttr>;` to resolve the references and declare the attributes at once (SIMD-group
        // ones too).
        var aliasLines: [String] = []
        for sp in parsed.semanticParams {
            aliasLines.append("\(sp.type) \(sp.name) = \(sp.semantic);")
        }
        let source = (aliasLines + [bodyClean]).joined(separator: "\n")

        return MLXFast.metalKernel(
            name: parsed.entry,
            inputNames: inputNames,
            outputNames: outputNames,
            source: source,
            header: parsed.header,
            atomicOutputs: atomicOutputs
        )
    }

    // MARK: - Resource loading

    private static func loadResource(name: String, extension ext: String, bundle: Bundle) throws -> String {
        if let url = bundle.url(forResource: name, withExtension: ext, subdirectory: "Slang") {
            return try String(contentsOf: url, encoding: .utf8)
        }
        // Depending on the SwiftPM bundle layout, files may sit at the root.
        if let url = bundle.url(forResource: name, withExtension: ext) {
            return try String(contentsOf: url, encoding: .utf8)
        }
        // Bundle lookup can get multi-extension names like `name.refl.json` wrong, so search by
        // hand.
        if let root = bundle.resourceURL {
            let candidate = root.appendingPathComponent("Slang/\(name).\(ext)")
            if FileManager.default.fileExists(atPath: candidate.path) {
                return try String(contentsOf: candidate, encoding: .utf8)
            }
            let flat = root.appendingPathComponent("\(name).\(ext)")
            if FileManager.default.fileExists(atPath: flat.path) {
                return try String(contentsOf: flat, encoding: .utf8)
            }
        }
        throw Error.resourceNotFound("\(name).\(ext)")
    }

    // MARK: - .metal parsing

    struct Parsed {
        let header: String
        let entry: String
        let semanticParams: [SemanticParam]
        let bufferParams: [BufferParam]
        let body: String
    }

    struct SemanticParam {
        let type: String
        let name: String
        let semantic: String
    }

    struct BufferParam {
        let type: String
        let name: String
        let bufferIndex: Int
    }

    static func parseMetal(_ source: String) throws -> Parsed {
        guard let kw = source.range(of: "[[kernel]]") else {
            throw Error.parseFailed("`[[kernel]]` not found")
        }
        let rawHeader = String(source[..<kw.lowerBound])
        let header = sanitizeHeader(rawHeader)

        // After `[[kernel]]` comes `void <entry>(<sig>) { <body> }`.
        let afterKw = source[kw.upperBound...]
        guard let voidR = afterKw.range(of: "void") else {
            throw Error.parseFailed("`void` after [[kernel]] not found")
        }
        let afterVoid = afterKw[voidR.upperBound...]

        guard let openParen = afterVoid.firstIndex(of: "(") else {
            throw Error.parseFailed("missing `(` after entry name")
        }
        let entry = afterVoid[..<openParen].trimmingCharacters(in: .whitespacesAndNewlines)

        var depth = 1
        var i = afterVoid.index(after: openParen)
        while i < afterVoid.endIndex {
            let c = afterVoid[i]
            if c == "(" { depth += 1 }
            else if c == ")" {
                depth -= 1
                if depth == 0 { break }
            }
            i = afterVoid.index(after: i)
        }
        guard depth == 0 else { throw Error.parseFailed("unbalanced `(` in kernel signature") }
        let sigContent = String(afterVoid[afterVoid.index(after: openParen)..<i])
        let (semantics, buffers) = try parseSignature(sigContent)

        let afterSig = afterVoid[afterVoid.index(after: i)...]
        guard let openBrace = afterSig.firstIndex(of: "{") else {
            throw Error.parseFailed("kernel body `{` not found")
        }
        var bd = 1
        var b = afterSig.index(after: openBrace)
        while b < afterSig.endIndex {
            let c = afterSig[b]
            if c == "{" { bd += 1 }
            else if c == "}" {
                bd -= 1
                if bd == 0 { break }
            }
            b = afterSig.index(after: b)
        }
        guard bd == 0 else { throw Error.parseFailed("unbalanced `{` in body") }
        let body = String(afterSig[afterSig.index(after: openBrace)..<b])

        return Parsed(
            header: header,
            entry: entry,
            semanticParams: semantics,
            bufferParams: buffers,
            body: body
        )
    }

    private static func parseSignature(_ s: String) throws -> ([SemanticParam], [BufferParam]) {
        // Split on depth-0 `,` (to handle `[[...]]` and parenthesized types).
        var params: [String] = []
        var depth = 0
        var current = ""
        for c in s {
            if c == "(" || c == "[" { depth += 1 }
            else if c == ")" || c == "]" { depth -= 1 }
            if c == "," && depth == 0 {
                params.append(current)
                current = ""
            } else {
                current.append(c)
            }
        }
        let last = current.trimmingCharacters(in: .whitespacesAndNewlines)
        if !last.isEmpty { params.append(current) }

        let bufferRegex = try NSRegularExpression(pattern: #"\[\[buffer\((\d+)\)\]\]"#)

        var semantics: [SemanticParam] = []
        var buffers: [BufferParam] = []
        for raw in params {
            let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty { continue }
            let range = NSRange(trimmed.startIndex..<trimmed.endIndex, in: trimmed)

            if let m = bufferRegex.firstMatch(in: trimmed, range: range),
               let idxR = Range(m.range(at: 1), in: trimmed),
               let attrR = Range(m.range, in: trimmed)
            {
                let bufferIndex = Int(trimmed[idxR]) ?? -1
                let before = trimmed[..<attrR.lowerBound].trimmingCharacters(in: .whitespacesAndNewlines)
                let parts = before.split(whereSeparator: { $0.isWhitespace })
                guard let nameTok = parts.last else {
                    throw Error.parseFailed("buffer param missing name: \(trimmed)")
                }
                let name = String(nameTok)
                let type = parts.dropLast().joined(separator: " ")
                buffers.append(BufferParam(type: type, name: name, bufferIndex: bufferIndex))
            } else if let attrStart = trimmed.range(of: "[[")?.lowerBound,
                      let attrEnd = trimmed.range(of: "]]")?.upperBound
            {
                let semantic = trimmed[trimmed.index(attrStart, offsetBy: 2)..<trimmed.index(attrEnd, offsetBy: -2)]
                let semStr = String(semantic).trimmingCharacters(in: .whitespacesAndNewlines)
                let before = trimmed[..<attrStart].trimmingCharacters(in: .whitespacesAndNewlines)
                let parts = before.split(whereSeparator: { $0.isWhitespace })
                guard let nameTok = parts.last else {
                    throw Error.parseFailed("semantic param missing name: \(trimmed)")
                }
                let name = String(nameTok)
                let type = parts.dropLast().joined(separator: " ")
                semantics.append(SemanticParam(type: type, name: name, semantic: semStr))
            } else {
                throw Error.parseFailed("unrecognised parameter: \(trimmed)")
            }
        }
        return (semantics, buffers)
    }

    /// Strips references through Slang's `KernelContext_*`. MLX declares inputs as `device const
    /// T*`, so assignments into the context become const mismatches, and since the struct is also
    /// removed from the header the references dangle. Collects FIELD → PARAM from the assignment
    /// lines, removes those lines and the declaration, and replaces the remaining
    /// `(&kernelContext_N)->FIELD`. Also removes `#line` (it confuses Metal once MLX re-attaches
    /// the signature).
    static func stripKernelContext(_ body: String) -> String {
        // The RHS can be an expression like `&sg_1` (address-of a threadgroup array), so capture up
        // to `;` with `[^;]+` (`\w+` would miss it and leave undefined references).
        let initPattern = try! NSRegularExpression(
            pattern: #"\(&kernelContext_\d+\)->(\w+)\s*=\s*([^;]+);"#
        )
        let derefPattern = try! NSRegularExpression(
            pattern: #"\(&kernelContext_\d+\)->(\w+)"#
        )

        // Pass 1: collect FIELD → PARAM from the assignment lines.
        var mapping: [String: String] = [:]
        let allLines = body.split(separator: "\n", omittingEmptySubsequences: false)
        for line in allLines {
            let str = String(line)
            let range = NSRange(str.startIndex..<str.endIndex, in: str)
            if let m = initPattern.firstMatch(in: str, range: range),
               let fieldR = Range(m.range(at: 1), in: str),
               let paramR = Range(m.range(at: 2), in: str)
            {
                mapping[String(str[fieldR])] =
                    String(str[paramR]).trimmingCharacters(in: .whitespaces)
            }
        }

        // Pass 2: drop unneeded lines and replace references.
        var out: [String] = []
        for line in allLines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("#line") { continue }
            if trimmed.hasPrefix("thread KernelContext") { continue }
            if trimmed.hasPrefix("(&kernelContext_") && trimmed.contains("=") {
                continue
            }

            var lineStr = String(line)
            let nsRange = NSRange(lineStr.startIndex..<lineStr.endIndex, in: lineStr)
            let matches = derefPattern.matches(in: lineStr, range: nsRange)
            // Replace from the back to keep earlier ranges valid.
            for m in matches.reversed() {
                guard let fullRng = Range(m.range, in: lineStr),
                      let fieldRng = Range(m.range(at: 1), in: lineStr)
                else { continue }
                let field = String(lineStr[fieldRng])
                guard let param = mapping[field] else { continue }
                lineStr.replaceSubrange(fullRng, with: param)
            }
            out.append(lineStr)
        }
        return out.joined(separator: "\n")
    }

    /// Removes `#line` and the `KernelContext_*` struct from the header (the struct is no longer
    /// needed, and `#line` confuses Metal's preprocessor once MLX re-attaches the signature).
    static func sanitizeHeader(_ raw: String) -> String {
        var out: [String] = []
        var skippingStruct = false
        var braceDepth = 0
        for line in raw.split(separator: "\n", omittingEmptySubsequences: false) {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if skippingStruct {
                for c in trimmed {
                    if c == "{" { braceDepth += 1 }
                    else if c == "}" {
                        braceDepth -= 1
                        if braceDepth == 0 {
                            skippingStruct = false
                            break
                        }
                    }
                }
                continue
            }
            if trimmed.hasPrefix("#line") { continue }
            if trimmed.hasPrefix("struct KernelContext") {
                skippingStruct = true
                braceDepth = 0
                for c in trimmed {
                    if c == "{" { braceDepth += 1 }
                    else if c == "}" {
                        braceDepth -= 1
                        if braceDepth == 0 {
                            skippingStruct = false
                            break
                        }
                    }
                }
                continue
            }
            out.append(String(line))
        }
        return out.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Reflection JSON

    struct Reflection: Decodable {
        struct Param: Decodable {
            struct Binding: Decodable {
                let index: Int
            }
            struct TypeDecl: Decodable {
                let access: String?
            }
            let name: String
            let binding: Binding
            let type: TypeDecl
        }
        let parameters: [Param]
    }

    static func parseReflection(_ json: String) throws -> Reflection {
        guard let data = json.data(using: .utf8) else {
            throw Error.decodeFailed("reflection JSON not UTF-8")
        }
        do {
            return try JSONDecoder().decode(Reflection.self, from: data)
        } catch {
            throw Error.decodeFailed(String(describing: error))
        }
    }
}
