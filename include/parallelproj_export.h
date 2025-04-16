#pragma once

#if defined(_WIN32)
#if defined(PARALLELPROJ_EXPORTS)
#define PARALLELPROJ_API __declspec(dllexport)
#else
#define PARALLELPROJ_API __declspec(dllimport)
#endif
#else
#define PARALLELPROJ_API
#endif
