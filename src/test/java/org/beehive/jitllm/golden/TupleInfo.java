package org.beehive.jllm.golden;

import uk.ac.manchester.tornado.api.TornadoDeviceMap;
import uk.ac.manchester.tornado.api.common.TornadoDevice;

/**
 * The pinned tuple a golden is valid on: device, driver, TornadoVM version, backend and the build
 * flags that change numerics. Recorded into every golden's metadata and re-read when comparing, so
 * a run on a different machine downgrades instead of producing a false failure.
 */
public final class TupleInfo {

    private TupleInfo() {}

    /**
     * The installed SDK's version. {@code Package.getImplementationVersion()} is null when
     * TornadoVM is loaded from the module path, so the authoritative source is {@code
     * $TORNADOVM_HOME/etc/tornado.release}, which is what {@code tornado --version} reads.
     */
    public static String tornadoVmVersion() {
        String home = System.getenv("TORNADOVM_HOME");
        if (home != null && !home.isBlank()) {
            java.nio.file.Path release = java.nio.file.Paths.get(home, "etc", "tornado.release");
            if (java.nio.file.Files.isRegularFile(release)) {
                try {
                    for (String line : java.nio.file.Files.readAllLines(release)) {
                        if (line.startsWith("version=")) {
                            return line.substring("version=".length()).trim();
                        }
                    }
                } catch (java.io.IOException ignored) {
                    // fall through to the package/property fallbacks
                }
            }
        }
        Package p = TornadoDeviceMap.class.getPackage();
        String v = p == null ? null : p.getImplementationVersion();
        if (v != null && !v.isBlank()) {
            return v;
        }
        return System.getProperty("tornado.version", "unknown");
    }

    public static TornadoDevice defaultDevice() {
        try {
            TornadoDeviceMap map = new TornadoDeviceMap();
            if (map.getNumBackends() == 0) {
                return null;
            }
            return map.getAllBackends().get(0).getDevice(0);
        } catch (RuntimeException | Error e) {
            return null;
        }
    }

    /**
     * The physical GPU name. {@code TornadoDevice.getDeviceName()} returns the Tornado device id
     * ("cuda-0-0"), which is identical on different hardware and so useless for pinning a tuple.
     */
    public static String deviceName() {
        TornadoDevice d = defaultDevice();
        return d == null ? "" : d.getPhysicalDevice().getDeviceName();
    }

    public static String backend() {
        TornadoDevice d = defaultDevice();
        return d == null ? "" : d.getTornadoVMBackend().name();
    }

    /**
     * There is no driver-version accessor on {@code TornadoDevice}, so the platform name plus the
     * device's OpenCL C version stand in as the driver half of the tuple.
     */
    public static String driver() {
        TornadoDevice d = defaultDevice();
        return d == null ? "" : d.getPlatformName() + " / " + d.getDeviceOpenCLCVersion();
    }

    /** True when a TornadoVM device is actually usable in this JVM. */
    public static boolean acceleratorPresent() {
        return defaultDevice() != null;
    }
}
