/*
 * Copyright (c) 2024-2026, APT Group, Department of Computer Science,
 * The University of Manchester.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
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
