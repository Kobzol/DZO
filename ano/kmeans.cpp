#include "stdafx.h"
#include "kmeans.h"

#include <ctime>

std::vector<std::vector<Feature>> KMeans::segment(const std::vector<Feature>& features, size_t k)
{
	srand((unsigned int) time(nullptr));
	std::vector<std::pair<Feature, std::vector<int>>> clusters;
	
	// Fisher-Yates
	std::vector<int> indexes;
	for (int i = 0; i < features.size(); i++)
	{
		indexes.push_back(i);
	}
	for (size_t i = 0; i < k; i++)
	{
		int pos = rand() % (indexes.size() - i);
		clusters.push_back({ features[indexes[pos]], {} });
		std::swap(indexes[pos], indexes[indexes.size() - i - 1]);
	}

	for (int i = 0; i < 1000; i++)
	{
		// reset objects
		for (int i = 0; i < clusters.size(); i++)
		{
			clusters[i].second.clear();
		}

		// find cluster for each object
		for (int f = 0; f < features.size(); f++)
		{
			int clusterIndex = 0;
			for (int c = 1; c < clusters.size(); c++)
			{
				if (clusters[c].first.distance(features[f]) < clusters[clusterIndex].first.distance(features[f]))
				{
					clusterIndex = c;
				}
			}

			clusters[clusterIndex].second.push_back(f);
		}

		double delta = 0.0;
		// reassign clusters
		for (int i = 0; i < clusters.size(); i++)
		{
			Feature newCenter;
			for (int f = 0; f < clusters[i].second.size(); f++)
			{
				newCenter.f1 += features[clusters[i].second[f]].f1;
				newCenter.f2 += features[clusters[i].second[f]].f2;
			}
			newCenter.f1 /= clusters[i].second.size();
			newCenter.f2 /= clusters[i].second.size();

			delta = std::max(delta, newCenter.distance(clusters[i].first));
			clusters[i].first = newCenter;
		}

		std::cerr << "Delta: " << delta << std::endl;
		if (delta < 0.001f)
		{
			break;
		}
	}

	// set result
	std::vector<std::vector<Feature>> result;
	for (int i = 0; i < clusters.size(); i++)
	{
		result.push_back({});
		for (int f = 0; f < clusters[i].second.size(); f++)
		{
			result[i].push_back(features[clusters[i].second[f]]);
		}
	}

	return result;
}
