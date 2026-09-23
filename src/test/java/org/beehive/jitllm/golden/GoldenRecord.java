package org.beehive.jitllm.golden;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The full logits are ~31.8 MiB per configuration (128256 vocab x 65 rows x 4 B), which is far too
 * large to commit. Instead each compared row is reduced to a SHA-256 — bit-level drift in any row
 * still fails — and the <b>final</b> row is kept verbatim as {@code.f32le} so a failure can be
 * diffed numerically and so a non-pinned tuple has real values to apply the parity tolerance to.
 *
 * <p>Files per golden directory:
 *
 * <pre>
 *   metadata.json     tuple, prompt, fixture sha256, generating commit
 *   row-hashes.txt    one SHA-256 per compared logits row, in order
 *   token-ids.txt     the emitted token ids, in order
 *   final-row.f32le   the last compared logits row, little-endian float32
 * </pre>
 */
public final class GoldenRecord {

    public final Map<String, String> metadata;
    public final List<String> rowHashes;
    public final List<Integer> tokenIds;
    public final float[] finalRow;

    public GoldenRecord(
            Map<String, String> metadata,
            List<String> rowHashes,
            List<Integer> tokenIds,
            float[] finalRow) {
        this.metadata = metadata;
        this.rowHashes = rowHashes;
        this.tokenIds = tokenIds;
        this.finalRow = finalRow;
    }

    /**
     * SHA-256 over the raw little-endian float32 bytes of one logits row.
     *
     * <p>Hashing the raw bit patterns (not formatted text) keeps the comparison exactly as strict
     * as {@code Float.floatToRawIntBits} equality over the whole row.
     */
    public static String hashRow(float[] row) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            ByteBuffer bb = ByteBuffer.allocate(row.length * 4).order(ByteOrder.LITTLE_ENDIAN);
            for (float v : row) {
                bb.putInt(Float.floatToRawIntBits(v));
            }
            md.update(bb.array());
            return HexFormat.of().formatHex(md.digest());
        } catch (java.security.NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 unavailable", e);
        }
    }

    public void write(Path dir) throws IOException {
        Files.createDirectories(dir);

        StringBuilder json = new StringBuilder("{\n");
        int i = 0;
        for (Map.Entry<String, String> e : metadata.entrySet()) {
            json.append("  \"").append(e.getKey()).append("\": ");
            String v = e.getValue();
            boolean numericOrBool = v.equals("true") || v.equals("false") || v.matches("-?\\d+");
            json.append(numericOrBool ? v : "\"" + escape(v) + "\"");
            json.append(++i < metadata.size() ? ",\n" : "\n");
        }
        json.append("}\n");
        Files.writeString(dir.resolve("metadata.json"), json.toString(), StandardCharsets.UTF_8);

        Files.write(dir.resolve("row-hashes.txt"), rowHashes);
        List<String> ids = new ArrayList<>();
        tokenIds.forEach(t -> ids.add(Integer.toString(t)));
        Files.write(dir.resolve("token-ids.txt"), ids);

        ByteBuffer bb = ByteBuffer.allocate(finalRow.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (float v : finalRow) {
            bb.putInt(Float.floatToRawIntBits(v));
        }
        Files.write(dir.resolve("final-row.f32le"), bb.array());
    }

    public static GoldenRecord read(Path dir) throws IOException {
        Map<String, String> meta = new LinkedHashMap<>();
        for (String line : Files.readAllLines(dir.resolve("metadata.json"))) {
            String s = line.trim();
            int colon = s.indexOf("\":");
            if (!s.startsWith("\"") || colon < 0) {
                continue;
            }
            String k = s.substring(1, colon);
            String v = s.substring(colon + 2).trim();
            if (v.endsWith(",")) {
                v = v.substring(0, v.length() - 1);
            }
            if (v.startsWith("\"") && v.endsWith("\"")) {
                v = v.substring(1, v.length() - 1);
            }
            meta.put(k, v);
        }
        List<String> hashes = Files.readAllLines(dir.resolve("row-hashes.txt"));
        List<Integer> ids = new ArrayList<>();
        for (String s : Files.readAllLines(dir.resolve("token-ids.txt"))) {
            if (!s.isBlank()) {
                ids.add(Integer.parseInt(s.trim()));
            }
        }
        byte[] raw = Files.readAllBytes(dir.resolve("final-row.f32le"));
        ByteBuffer bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
        float[] row = new float[raw.length / 4];
        for (int i = 0; i < row.length; i++) {
            row[i] = Float.intBitsToFloat(bb.getInt());
        }
        return new GoldenRecord(meta, hashes, ids, row);
    }

    private static String escape(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n");
    }
}
