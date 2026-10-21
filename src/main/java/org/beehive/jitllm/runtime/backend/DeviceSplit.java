package org.beehive.jitllm.runtime.backend;

import java.util.Arrays;
import java.util.List;
import java.util.Optional;

// @formatter:off
/**
 * A model's layers split across several devices, one pipeline stage per device, as llama.cpp's
 * {@code --split-mode layer} does it.
 *
 * <p>Not {@code ExecutionPolicy}: which device holds which layers decides where every weight and
 * cache array is allocated, and a value that sizes or places device arrays belongs to whoever
 * allocates. It is resolved once, where the plan is built, from the three properties the launcher
 * and {@code Options} set ({@code --devices}, {@code --tensor-split}, {@code --split-transport}).
 *
 * @param devices one device per stage, as the backend's own {@code backend:device} indices
 * @param shares each stage's share of the layers, in proportion (equal when empty)
 * @param transport how the hidden state moves from one stage to the next
 */
// @formatter:on
public record DeviceSplit(List<String> devices, List<Double> shares, Transport transport) {

    /** How the hidden state crosses between devices. */
    public enum Transport {
        /** Device to device with NCCL; needs a build with the {@code nccl} profile. */
        NCCL,
        /** Through host memory, on the calling thread. */
        HOST
    }

    /** {@code --devices}: e.g. {@code 0:0,0:1}. */
    public static final String DEVICES_PROPERTY = "jitllm.pipeline.devices";

    /** {@code --tensor-split}: e.g. {@code 41,39}. */
    public static final String SHARES_PROPERTY = "jitllm.pipeline.split";

    /** {@code --split-transport}: {@code nccl} (the default) or {@code host}. */
    public static final String TRANSPORT_PROPERTY = "jitllm.pipeline.transport";

    public DeviceSplit {
        devices = List.copyOf(devices);
        shares = List.copyOf(shares);
        if (devices.isEmpty()) {
            throw new IllegalArgumentException("a device split needs at least one device");
        }
        if (!shares.isEmpty() && shares.size() != devices.size()) {
            throw new IllegalArgumentException(
                    "--tensor-split has "
                            + shares.size()
                            + " entries for "
                            + devices.size()
                            + " devices");
        }
        for (double share : shares) {
            if (!(share > 0)) {
                throw new IllegalArgumentException("--tensor-split entries must be > 0");
            }
        }
        for (String device : devices) {
            if (!device.matches("\\d+:\\d+")) {
                throw new IllegalArgumentException(
                        "--devices entries are backend:device indices, not " + device);
            }
        }
    }

    /** The split the properties ask for, or empty when no devices are named. */
    public static Optional<DeviceSplit> fromSystemProperties() {
        String devices = System.getProperty(DEVICES_PROPERTY);
        if (devices == null || devices.isBlank()) {
            return Optional.empty();
        }
        String shares = System.getProperty(SHARES_PROPERTY);
        String transport = System.getProperty(TRANSPORT_PROPERTY, "nccl");
        return Optional.of(
                new DeviceSplit(
                        Arrays.stream(devices.split(",")).map(String::trim).toList(),
                        shares == null || shares.isBlank()
                                ? List.of()
                                : Arrays.stream(shares.split(","))
                                        .map(s -> Double.parseDouble(s.trim()))
                                        .toList(),
                        switch (transport) {
                            case "nccl" -> Transport.NCCL;
                            case "host" -> Transport.HOST;
                            default ->
                                    throw new IllegalArgumentException(
                                            "--split-transport must be nccl or host, not "
                                                    + transport);
                        }));
    }

    /** How many devices a split from the properties uses: 1 when none is asked for. */
    public static int requestedDeviceCount() {
        return fromSystemProperties().map(DeviceSplit::stages).orElse(1);
    }

    public int stages() {
        return devices.size();
    }

    /**
     * Stage boundaries for {@code layers} layers: stage {@code s} runs {@code [bounds[s], bounds[s
     * + 1])}. Layers are shared out in proportion to {@link #shares}, every stage keeping at least
     * one; inner boundaries then move to the nearest multiple of {@code alignment}, for a family
     * whose graphs group layers.
     */
    public int[] layerBounds(int layers, int alignment) {
        int stages = stages();
        if (layers < stages * alignment) {
            throw new IllegalArgumentException(
                    layers + " layers cannot be split across " + stages + " devices");
        }
        double[] share = new double[stages];
        for (int s = 0; s < stages; s++) {
            share[s] = shares.isEmpty() ? 1.0 : shares.get(s);
        }
        double total = Arrays.stream(share).sum();
        int[] bounds = new int[stages + 1];
        double cumulative = 0;
        for (int s = 0; s < stages; s++) {
            cumulative += share[s];
            int end = (int) Math.round(layers * cumulative / total);
            // At least one layer per stage, and room left for the stages after this one.
            end = Math.max(end, bounds[s] + 1);
            end = Math.min(end, layers - (stages - 1 - s));
            bounds[s + 1] = end;
        }
        bounds[stages] = layers;
        return alignment == 1 ? bounds : align(bounds, alignment);
    }

    /**
     * {@code bounds} with every inner boundary on a multiple of {@code group}, one group or more
     * each.
     */
    private static int[] align(int[] bounds, int group) {
        int stages = bounds.length - 1;
        int layers = bounds[stages];
        int[] aligned = bounds.clone();
        for (int s = 1; s < stages; s++) {
            int b = Math.round(bounds[s] / (float) group) * group;
            b = Math.max(b, aligned[s - 1] + group);
            b = Math.min(b, layers - (stages - s) * group);
            if (b <= aligned[s - 1] || b >= layers) {
                throw new IllegalArgumentException(
                        layers
                                + " layers cannot be split into "
                                + stages
                                + " stages on multiples of "
                                + group);
            }
            aligned[s] = b;
        }
        return aligned;
    }
}
