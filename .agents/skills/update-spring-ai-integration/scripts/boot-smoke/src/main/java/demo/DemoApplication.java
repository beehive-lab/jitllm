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
package demo;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.tool.annotation.Tool;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;

@SpringBootApplication
public class DemoApplication {

	public static void main(String[] args) {
		SpringApplication.run(DemoApplication.class, args).close();
	}

	static class Weather {

		@Tool(description = "Returns the current weather in a city")
		String getWeather(String city) {
			System.out.println("[tool] getWeather(" + city + ")");
			return "It is sunny and 21 degrees Celsius in " + city + ".";
		}

	}

	@Bean
	CommandLineRunner demo(ChatClient.Builder builder) {
		return args -> {
			ChatClient client = builder.build();
			ChatResponse response = client.prompt().user("Who are you? One sentence.").call().chatResponse();
			System.out.println("[call] " + response.getResult().getOutput().getText());
			System.out.println("[meta] model=" + response.getMetadata().getModel() + " usage="
					+ response.getMetadata().getUsage() + " on-gpu=" + response.getMetadata().get("on-gpu")
					+ " decode tok/s=" + response.getMetadata().get("generated-tokens-per-second"));
			System.out.print("[stream] ");
			client.prompt().user("Count from 1 to 10.").stream().content().doOnNext(System.out::print).blockLast();
			System.out.println();
			System.out.println("[tools] " + client.prompt()
				.user("What is the weather in Munich right now? Use the tool.")
				.tools(new Weather())
				.call()
				.content());
		};
	}

}
