#pragma once

#include <vector>

#include "segmentation.h"

class KMeans
{
public:
	std::vector<std::vector<Feature>> segment(const std::vector<Feature>& features, size_t k);
};
