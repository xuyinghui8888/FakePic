#ifndef __CGPBASE_H__
#define __CGPBASE_H__

#include <set>
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <strstream>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <random>
#include <algorithm>
#include <random>
#include <numeric>
#include <iterator>
#include <memory>
#include <array>
#include <type_traits>

#include <boost/filesystem.hpp> 

#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "DefinedMacro.h"
#include "SafeGuard.h"

#ifndef _MINI
#include <process.h>
#include <gtest/gtest.h>
#include "../GlogHelper/GLogHelper.h"
#else
#include "../TranslateCGP.h"
#endif

#endif