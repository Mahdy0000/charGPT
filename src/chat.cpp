// chat.cpp
//
// Interactive terminal inference program. Loads a trained model from model.bin,
// accepts text prompts from the user, and generates a continuation using the trained
// GPT model.
//
// The user's input is encoded into token IDs by the same CharTokenizer that was used
// during training. The generate() method runs the forward pass repeatedly, sampling
// the next token from the final row of the logit matrix at each step. The new token is
// appended to the context and the process repeats until gen_length tokens have been
// produced. The output tokens (excluding the prompt tokens) are then decoded back into
// a string and printed.
//
// Temperature controls the sharpness of the sampling distribution. At temperature 1.0
// the distribution is the raw softmax output. At temperature 0.5 the distribution is
// sharper and the model favours its highest-probability predictions. At temperature 2.0
// the distribution is flatter and the output is more varied and unpredictable. Values
// below 0.3 tend to produce repetitive loops.

#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include "model.hpp"

int main(int argc, char* argv[]) {
    // Defaults. The model path is positional; temperature and length accept flags.
    std::string model_path = "model.bin";
    float temperature = 0.8f;
    int gen_length = 200;

    // Parse command-line arguments. Unrecognised arguments are treated as the model path.
    // This allows the most common invocation "chat.exe model.bin" to work without flags.
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-t") && i+1 < argc) temperature = (float)std::atof(argv[++i]);
        else if ((arg == "-l") && i+1 < argc) gen_length = std::atoi(argv[++i]);
        else model_path = arg;
    }

    // Load the model and tokenizer from the binary file written by model.save().
    // The tokenizer vocabulary is embedded in the file so no training data is needed.
    std::cout << "Loading " << model_path << "..." << std::endl;
    CharTokenizer tokenizer;
    GPT model = GPT::load(model_path, tokenizer);
    if (model.vocab_size == 0) { std::cerr << "Failed. Train first." << std::endl; return 1; }

    std::cout << "  Temp: " << temperature << " | Length: " << gen_length << "\n" << std::endl;
    std::cout << "  Commands: /quit /temp <n> /len <n> /sample\n" << std::endl;

    std::string input;
    while (true) {
        std::cout << "> ";
        std::getline(std::cin, input);

        // Handle end-of-file (Ctrl+D on Linux, Ctrl+Z on Windows) and the quit command.
        if (std::cin.eof() || input == "/quit") { std::cout << "Bye!" << std::endl; break; }

        // /temp adjusts the sampling temperature. Values are clamped to [0.1, 2.0]
        // to prevent degenerate outputs from extreme settings.
        if (input.substr(0,5) == "/temp") {
            temperature = std::max(0.1f, std::min(2.0f, (float)std::atof(input.substr(6).c_str())));
            std::cout << "  Temp: " << temperature << std::endl; continue;
        }

        // /len adjusts the number of new tokens generated per response. Clamped to
        // [10, 2000] to prevent accidentally freezing the terminal with a huge output.
        if (input.substr(0,4) == "/len") {
            gen_length = std::max(10, std::min(2000, std::atoi(input.substr(5).c_str())));
            std::cout << "  Len: " << gen_length << std::endl; continue;
        }

        // /sample generates from a blank newline context, useful for seeing what the
        // model produces without any prompt conditioning.
        if (input == "/sample") input = "\n";

        if (input.empty()) continue;

        // Encode the prompt, run generation, then print only the newly generated tokens.
        // The prompt tokens are included in the returned context so they must be sliced off.
        auto tokens = tokenizer.encode(input);
        if (tokens.empty()) continue;
        auto out = model.generate(tokens, gen_length, temperature);
        std::cout << tokenizer.decode(std::vector<int>(out.begin()+tokens.size(), out.end()))
                  << "\n" << std::endl;
    }
    return 0;
}
