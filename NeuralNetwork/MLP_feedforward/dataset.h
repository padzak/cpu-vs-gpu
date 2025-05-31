#pragma once

#include "config.h"
#include "lodepng.h"

#include <filesystem>
#include <random>
#include <vector>


struct Sample {
    std::vector<float> pixels;   // length INPUT
    int                 label;   // 0-61
};

inline std::vector<Sample> load_dataset(const std::string& root)
{
    std::vector<Sample> data;
    std::vector<std::string> subdirs;

    for (auto& entry : std::filesystem::directory_iterator(root))
        if (entry.is_directory())
            subdirs.push_back(entry.path().string());

    // map directory name → label index
    std::unordered_map<std::string,int> label_of;
    auto add = [&](const std::string& name) {
        static int next = 0;
        label_of[name] = next++;
    };
    for (std::string n : {"0","1","2","3","4","5","6","7","8","9"}) add(n);
    for (char c='a'; c<='z'; ++c) add(std::string(1,c));
    for (char c='A'; c<='Z'; ++c) add(std::string(1,c) + "_caps");    // matches “<letter>_caps”

    for (const auto& dir : subdirs) {
        std::string base = std::filesystem::path(dir).filename().string();
        if (!label_of.count(base)) continue;               // skip unknown dirs
        int lbl = label_of[base];

        for (auto& imgPath : std::filesystem::directory_iterator(dir)) {
            if (imgPath.path().extension() != ".png") continue;
            std::vector<unsigned char> png, rgba;
            lodepng::load_file(png, imgPath.path().string());
            unsigned w,h;
            lodepng::decode(rgba, w, h, png);

            // very tiny nearest-neighbour resize to 32×32 gray for simplicity
            std::vector<float> px(INPUT);
            for(int y=0;y<IMG_H;++y)
                for(int x=0;x<IMG_W;++x){
                    unsigned srcY = y*h/IMG_H, srcX = x*w/IMG_W;
                    unsigned idx  = 4*(srcY*w + srcX);
                    unsigned char R = rgba[idx];
                    float gray = R / 255.f;                // assume R==G==B
                    px[y*IMG_W+x] = gray;
                }
            data.push_back({std::move(px), lbl});
        }
    }
    // shuffle once so batches are mixed
    std::mt19937 rng(42);
    std::shuffle(data.begin(), data.end(), rng);
    return data;
}
