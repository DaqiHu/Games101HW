#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "Eigen/src/Core/Matrix.h"
#include "rasterizer.hpp"


constexpr double MY_PI = 3.1415926;

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;

    // clang-format off
    translate <<
        1, 0, 0, -eye_pos[0],
        0, 1, 0, -eye_pos[1],
        0, 0, 1, -eye_pos[2],
        0, 0, 0, 1;
    // clang-format on

    view = translate * view;

    return view;
}

Eigen::Matrix4f get_model_matrix(float rot_z)
{
    Eigen::Matrix4f model;

    // clang-format off
    model <<
        cos(rot_z), -sin(rot_z), 0, 0,
        sin(rot_z), cos(rot_z), 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;
    // clang-format on

    return model;
}

Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio, float zNear, float zFar)
{
    float l, r, b, t, n, f;

    n = zNear;
    f = zFar;

    t = tan(eye_fov / 2) * abs(zNear);
    b = -t;
    r = aspect_ratio * t;
    l = -r;

    Eigen::Matrix4f PerspToOrtho, OrthoTranslation, OrthoScaling;

    // clang-format off
    PerspToOrtho <<
        n, 0, 0, 0,
        0, n, 0, 0,
        0, 0, n + f, -f * n,
        0, 0, 1, 0;

    OrthoTranslation <<
        1, 0, 0, -(r + l) / 2,
        0, 1, 0, -(t + b) / 2,
        0, 0, 1, -(n + f) / 2,
        0, 0, 0, 1;

    OrthoScaling <<
        2 / (r - l), 0, 0, 0,
        0, 2 / (t - b), 0, 0,
        0, 0, 2 / (n - f), 0,
        0, 0, 0, 1;
    // clang-format on

    return OrthoScaling * OrthoTranslation * PerspToOrtho;
}

auto get_image(float angle, const Eigen::Vector3f& eye_pos, const rst::pos_buf_id& pos_id, const rst::ind_buf_id& ind_id)
{
    rst::rasterizer r{700, 700};
    r.clear(rst::Buffers::Color | rst::Buffers::Depth);

    r.set_model(get_model_matrix(angle));
    r.set_view(get_view_matrix(eye_pos));
    r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

    r.draw(pos_id, ind_id, rst::Primitive::Triangle);
    cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
    image.convertTo(image, CV_8UC3, 1.0f);

    return image;
}

int main(int argc, const char** argv)
{
    float angle          = 0;
    bool command_line    = false;
    std::string filename = "output.png";

    if (argc >= 3)
    {
        command_line = true;
        angle        = std::stof(argv[2]); // -r by default
        if (argc == 4)
        {
            filename = std::string(argv[3]);
        }
        else
            return 0;
    }

    rst::rasterizer r(700, 700);

    Eigen::Vector3f eye_pos = {0, 0, 5};

    std::vector<Eigen::Vector3f> pos{{2, 0, -2}, {0, 2, -2}, {-2, 0, -2}};

    std::vector<Eigen::Vector3i> ind{{0, 1, 2}};

    auto pos_id = r.load_positions(pos);
    auto ind_id = r.load_indices(ind);

    int key         = 0;
    int frame_count = 0;

    if (command_line)
    {
        cv::imwrite(filename, get_image(angle, eye_pos, pos_id, ind_id));

        return 0;
    }

    while (key != 27)
    {
        cv::imshow("image", get_image(angle, eye_pos, pos_id, ind_id));
        key = cv::waitKey(10);

        std::cout << "frame count: " << frame_count++ << '\n';

        if (key == 'a')
        {
            angle += 10;
        }
        else if (key == 'd')
        {
            angle -= 10;
        }
    }

    return 0;
}
