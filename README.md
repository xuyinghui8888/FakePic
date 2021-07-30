# Introduction  
This code base is used for avatar generation playground.   
This is the code base for basic avatar generation.
# Dependencies
* opencv >= 4.0
* boost >= 1.60.0
* rttr >= 1.0.0
* mnn >=0.9.0


# Getting started
* server data access right
<http://gitlab.alibaba-inc.com/zhaohaiming.pt/server_data>

# Detailed information
## Data format  
Data loading && saving using binary or text, json, yml format to support for int, string, or cv Mat data format.

# Code Style
Coding guidelines are according to Google c++ guideline. Except for following cases:  
<https://google.github.io/styleguide/cppguide.html>  
function begins with small letters.
```
void getFunctionNames();
```
Namespace variable is variable: 
```
namespace CGP;
```
or
```
namespace Cgp;
```

enum class template:
```
enum class TensorType
{
    BFM_DEEP, //set for 3dmm deepFaceReconstruction based model
};
```

# Copyright

Copyright (c) 2019 Alibaba Group Holding Limited
Author: Zhao Haiming (Suyu)