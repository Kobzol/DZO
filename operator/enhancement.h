#pragma once

#include <cmath>

template <class T>
T simple_contrast_enhancement(T p, T min, T max, T scale = 1)
{
	return static_cast<T>(((p - min) / static_cast<float>(max - min)) * scale);
}
template <class T>
T gamma_correct(T p, double scale)
{
	return static_cast<T>(std::pow(static_cast<double>(p), scale));
}
