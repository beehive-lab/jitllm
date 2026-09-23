package org.beehive.jitllm.golden;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.format.ChatFormat;
import org.beehive.jitllm.model.loader.ModelLoader;

/**
 * Diagnostic: prompt token count, to build prompts whose length is an exact multiple of a chunk.
 */
public final class TokenCount {
    public static void main(String[] args) throws Exception {
        Model model =
                ModelLoader.loadModel(Path.of(System.getProperty("tc.model")), 512, true, false);
        System.out.println("config: " + model.configuration());
        ChatFormat cf = model.chatFormat();
        for (String p : System.getProperty("tc.prompts").split("\\|\\|")) {
            List<Integer> t = new ArrayList<>();
            if (model.shouldAddBeginOfText()) {
                t.add(cf.getBeginOfText());
            }
            t.addAll(cf.encodeMessage(new ChatFormat.Message(ChatFormat.Role.USER, p)));
            t.addAll(cf.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
            System.out.printf("%d tokens : %s%n", t.size(), p);
        }
    }

    private TokenCount() {}
}
