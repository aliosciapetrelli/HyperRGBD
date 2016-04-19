#include "hyperrgbdUtils.h"


#include <algorithm>

#ifndef SNAPPY_IS_NOT_INCLUDED
#include "snappy-c.h"
#endif


using namespace std;
using namespace cv;


#ifdef _MSC_VER
    #ifndef _CRT_SECURE_NO_WARNINGS
    #define _CRT_SECURE_NO_WARNINGS
    #endif
    #define NOMINMAX
    #include "windows.h"
#endif

#include <io.h>   // For access().
#include <sys/types.h>  // For stat().
#include <sys/stat.h>   // For stat().



bool HyperRGBD::ExistsFile(const string &filename_abs)
{
	#ifdef _MSC_VER
		return !(GetFileAttributes(filename_abs.c_str()) == INVALID_FILE_ATTRIBUTES); 			
	#else	
		#ifndef BOOST_IS_NOT_INCLUDED
			return boost::filesystem::exists( filename_abs.c_str() );
		#else		
			#ifdef __GNUC__
				//assume that on Linux the compiler is gcc 
				struct stat st;
				return (stat(filename_abs.c_str(), &st) == 0);
			#else			
				cout << "ERROR (FileExists): unknown environment";
				getchar();
				exit(-1);

				return true;

			#endif

		#endif

	#endif

	////old version (with explicit check if file is a directory)
	//if ( _access( filename, 0 ) == 0 )
	//{
	//	struct stat status;
	//	stat( filename, &status );

	//	if ( status.st_mode & S_IFDIR )
	//	{
	//		//The path you entered is a directory.
	//		return false;
	//	}

	//	return true;
	//}

	//return false;

}

void HyperRGBD::Prefix_Path(string &filename, const string &pathToPrefix)
{
	int nSlash = 0;
	if( (filename[0] == '\\') || (filename[0] == '/'))
	{
		nSlash++;
	}
	if( (pathToPrefix[pathToPrefix.size()-1] == '\\') || (pathToPrefix[pathToPrefix.size()-1] == '/'))
	{
		nSlash++;
	}
	if(nSlash == 0)
	{
		filename = pathToPrefix + "/" + filename;
		return;
	}
	if(nSlash == 1)
	{
		filename = pathToPrefix + filename;
		return;
	}
	if(nSlash == 2)
	{
		filename = pathToPrefix.substr(0, pathToPrefix.size()-1)  + filename;
		return;
	}

}


void HyperRGBD::Prefix_Path(vector< string > &vFilenames, const string &pathToPrefix)
{	
	for(size_t sa=0; sa<vFilenames.size(); sa++)
	{
		Prefix_Path(vFilenames[sa], pathToPrefix);
	}
}


size_t HyperRGBD::Sample_random(std::vector<bool> &vData, const float samplingFactor, const unsigned int seed)
{
	if(seed != std::numeric_limits<unsigned int>::max())
	{
		srand(seed);
	}

	size_t sampledSize =  size_t( samplingFactor * (float)(vData.size()) );

	if(sampledSize < 0)
	{
		throw runtime_error("ERROR (Sample_random): (sampledSize < 0)");
	}

	if(sampledSize > vData.size())
	{
		throw runtime_error("ERROR (Sample_random): (sampledSize > vData.size())");
	}

	for(size_t sa=0; sa<sampledSize; sa++)
	{
		vData[sa] = true;
	}
	for(size_t sa=sampledSize; sa<vData.size(); sa++)
	{
		vData[sa] = false;
	}
	
	random_shuffle(vData.begin(), vData.end(), NextRandom);

	return sampledSize;
}



std::string HyperRGBD::GetPathDir(const std::string &filename)
{
	string strFinal = filename;
	size_t posSlash = strFinal.rfind('/');
	size_t posBackSlash = strFinal.rfind('\\');
	if( (posSlash == string::npos) && (posBackSlash == string::npos) )
	{
		return ".";
	}

	if (posSlash == string::npos)
	{
		return strFinal.substr(0, posBackSlash);
	}

	if (posBackSlash == string::npos)
	{
		return strFinal.substr(0, posSlash);
	}

	return strFinal.substr(0, (posSlash > posBackSlash)?posSlash:posBackSlash);

}


string HyperRGBD::GetRelativeName(const string &filename)
{
	std::string strFinal = filename;
	size_t posSlash = strFinal.rfind('/');
	size_t posBackSlash = strFinal.rfind('\\');
	if( (posSlash == std::string::npos) && (posBackSlash == std::string::npos) )
	{
		return strFinal;
	}

	if (posSlash == std::string::npos)
	{
		return strFinal.substr(posBackSlash+1);
	}

	if (posBackSlash == std::string::npos)
	{
		return strFinal.substr(posSlash+1);
	}

	return strFinal.substr(Max(posSlash+1, posBackSlash+1));
}



bool HyperRGBD::FindDirsEndWith(const std::string & path, const std::string & endingString, std::vector<std::string> & foundDirs, bool getOnlyFileName, const int nMaxItems )
{
	foundDirs.clear();

#ifdef _MSC_VER
	// Required structs for searching for files and directories
	WIN32_FIND_DATA FindFileData;
	HANDLE hFind = INVALID_HANDLE_VALUE;

	// Build the file search string...
	char searchDir[2048] = {0};
	char fullpath[2048] = {0};

	// ...if it already is a path that ends with \ or /, add '*'...
	if(path.at(path.length() - 1) == '\\' || path.at(path.length() - 1) == '/')
	{
		_snprintf(searchDir, 2047, "%s*", path.c_str());
		_snprintf(fullpath, 2047, "%s", path.c_str()); // just copy path
	}
	// ...otherwise, add '\*' to the end of the path.
	else
	{
		_snprintf(searchDir, 2047, "%s\\*", path.c_str());
		_snprintf(fullpath, 2047, "%s/", path.c_str()); // copy path and add slash (required when building filenames)
	}

	// Find the first file in the directory.
	hFind = FindFirstFile(searchDir, &FindFileData);

	// If there is no file, return
	if (hFind == INVALID_HANDLE_VALUE)
	{
		return false;
	}

	// loop
	do
	{
		// Skip ".", ".."
		if(strcmp(FindFileData.cFileName, ".") == 0 || strcmp(FindFileData.cFileName, "..") == 0)
		{
			continue;
		}

		if(endingString.size() > std::string(FindFileData.cFileName).size())
		{
			continue;
		}

		// If a directory is found
		if((FindFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0)
		{
			// Store dir filename into the vector if it ends with the given string
			if(endingString.size() > 0)
			{
				if( endingString.size() <= std::string(FindFileData.cFileName).size() )
				{
					size_t pos = std::string(FindFileData.cFileName).rfind(endingString);

					if(pos >= 0 && pos == (std::string(FindFileData.cFileName).size() - endingString.size()) )
					{
						if(getOnlyFileName)
						{
							foundDirs.push_back(FindFileData.cFileName);
						}
						else
						{
							// Directory found: create path to the directory
							char dirPath[2048] = {0};
							_snprintf(dirPath, 2047, "%s%s", fullpath, FindFileData.cFileName);
							// Add it to vector of files found
							foundDirs.push_back(dirPath);
						}
					}
				}
			}
			else // Store dir filename if an ending string has not been provided
			{

				if(getOnlyFileName)
				{
					foundDirs.push_back(FindFileData.cFileName);
				}
				else
				{
				// Directory found: create path to the directory
				char dirPath[2048] = {0};
				_snprintf(dirPath, 2047, "%s%s", fullpath, FindFileData.cFileName);
				// Add it to vector of files found
					foundDirs.push_back(dirPath);
				}
			}
		}
	}
	// Loop while we find more files
	while(FindNextFile(hFind, &FindFileData) != 0);

	// Release
	FindClose(hFind);

	sort(foundDirs.begin(), foundDirs.end(), StringCompare_Smart_Incr());

	foundDirs.resize( Min(nMaxItems, (int)foundDirs.size() ) );

	return true;
#else // not _MSC_VER
    DIR* directory = opendir(path.c_str());
    if(directory)
    {
        string parent(path);
        if(parent[parent.length()-1] != '/')
            parent.append("/");

        struct dirent dirEntry;
        struct dirent* res = &dirEntry;
        while((readdir_r(directory, &dirEntry, &res) == 0) && (res)) // thread-safe
            if((dirEntry.d_type == DT_DIR) &&
                    (strncmp(dirEntry.d_name+(d_namlen-endingString.size()), endingString.c_str(), endingString.length()) == 0) &&
                    (strcmp(dirEntry.d_name, ".") != 0) &&
                    (strcmp(dirEntry.d_name, "..") != 0))
				if(getOnlyFileName)
				{
					foundFiles.push_back(dirEntry.d_name);
				}
				else
				{
					foundFiles.push_back(parent + dirEntry.d_name);
				}
        closedir(directory);

		sort(foundDirs.begin(), foundDirs.end(), StringCompare_Smart_Incr());

		foundDirs.resize( Min(nMaxItems, (int)foundDirs.size() ) );

        return true;
    }

    return false;
#endif // _MSC_VER
}



void HyperRGBD::CreateDir(const string &directory)
{
	#ifdef WIN32
		CreateDirectory(directory.c_str(), NULL);
	#elif defined (__GNUC__)
	/*  read/write/exe for owner
        read/exe for group owner
        read for other              */
        mkdir(directory.c_str(), S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH);
	#elif !defined (BOOST_IS_NOT_INCLUDED)
        create_directory(path(directory.c_str()));
	#endif
}


bool HyperRGBD::ExistsDir(const string &directory)
{
	if ( _access( directory.c_str(), 0 ) == 0 )
    {
        struct stat status;
        stat( directory.c_str(), &status );

        if ( status.st_mode & S_IFDIR )
        {
            return true;
        }

		//The path you entered is a file.
		return false;
    }

	return false;
}

void HyperRGBD::CreateFullPath(const string &directory)
{
	string strDirectory = directory;
	if( (strDirectory[strDirectory.size()-1] != ':') && (!ExistsDir(directory)) )
	{
		string subDir = GetPathDir(directory);
		CreateFullPath(subDir.c_str());
		CreateDir(directory);
	}
}

bool HyperRGBD::FindDirsStartWith(const std::string & path, const std::string & startingString, std::vector<std::string> & foundDirs, bool getOnlyFileName, const int nMaxItems )
{
	foundDirs.clear();

#ifdef _MSC_VER
	// Required structs for searching for files and directories
	WIN32_FIND_DATA FindFileData;
	HANDLE hFind = INVALID_HANDLE_VALUE;

	// Build the file search string...
	char searchDir[2048] = {0};
	char fullpath[2048] = {0};

	// ...if it already is a path that ends with \ or /, add '*'...
	if(path.at(path.length() - 1) == '\\' || path.at(path.length() - 1) == '/')
	{
		_snprintf(searchDir, 2047, "%s*", path.c_str());
		_snprintf(fullpath, 2047, "%s", path.c_str()); // just copy path
	}
	// ...otherwise, add '\*' to the end of the path.
	else
	{
		_snprintf(searchDir, 2047, "%s\\*", path.c_str());
		_snprintf(fullpath, 2047, "%s/", path.c_str()); // copy path and add slash (required when building filenames)
	}

	// Find the first file in the directory.
	hFind = FindFirstFile(searchDir, &FindFileData);

	// If there is no file, return
	if (hFind == INVALID_HANDLE_VALUE)
	{
		return false;
	}

	// loop
	do
	{
		// Skip ".", ".."
		if(strcmp(FindFileData.cFileName, ".") == 0 || strcmp(FindFileData.cFileName, "..") == 0)
		{
			continue;
		}
		// If a directory is found
		if((FindFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0)
		{
			// Store dir filename into the vector if it starts with the given string
			if(startingString.size() > 0)
			{
				if(std::string(FindFileData.cFileName).find(startingString) == 0)
				{
					if(getOnlyFileName)
					{
						foundDirs.push_back(FindFileData.cFileName);
					}
					else
					{
						// Directory found: create path to the directory
						char dirPath[2048] = {0};
						_snprintf(dirPath, 2047, "%s%s", fullpath, FindFileData.cFileName);
						// Add it to vector of files found
						foundDirs.push_back(dirPath);
					}
				}
			}
			else // Store dir filename if a starting string has not been provided
			{

				if(getOnlyFileName)
				{
					foundDirs.push_back(FindFileData.cFileName);
				}
				else
				{
				// Directory found: create path to the directory
				char dirPath[2048] = {0};
				_snprintf(dirPath, 2047, "%s%s", fullpath, FindFileData.cFileName);
				// Add it to vector of files found
					foundDirs.push_back(dirPath);
				}
			}
		}
	}
	// Loop while we find more files
	while(FindNextFile(hFind, &FindFileData) != 0);

	// Release
	FindClose(hFind);

	sort(foundDirs.begin(), foundDirs.end(), StringCompare_Smart_Incr());

	foundDirs.resize( Min(nMaxItems, (int)foundDirs.size() ) );

	return true;
#else // not _MSC_VER
    DIR* directory = opendir(path.c_str());
    if(directory)
    {
        string parent(path);
        if(parent[parent.length()-1] != '/')
            parent.append("/");

        struct dirent dirEntry;
        struct dirent* res = &dirEntry;
        while((readdir_r(directory, &dirEntry, &res) == 0) && (res)) // thread-safe
            if((dirEntry.d_type == DT_DIR) &&
                    (strncmp(dirEntry.d_name, startingString.c_str(), startingString.length()) == 0) &&
                    (strcmp(dirEntry.d_name, ".") != 0) &&
                    (strcmp(dirEntry.d_name, "..") != 0))
				if(getOnlyFileName)
				{
					foundFiles.push_back(dirEntry.d_name);
				}
				else
				{
					foundFiles.push_back(parent + dirEntry.d_name);
				}
        closedir(directory);

		sort(foundDirs.begin(), foundDirs.end(), StringCompare_Smart_Incr());
		foundDirs.resize( Min(nMaxItems, (int)foundDirs.size() ) );

        return true;
    }

    return false;
#endif // _MSC_VER
}


bool HyperRGBD::FindMaskBoundingBox(cv::Mat &mat, cv::Rect &boundingBox, const float sigmaMult, const unsigned char maskValue)
{
	vector<float> vX;
	vector<float> vY;


	for( int ro = 0; ro< mat.rows; ro++ )
	{
		for( int co = 0; co< mat.cols; co++ )
		{
			if(mat.at<unsigned char>(ro,co) == maskValue)
			{
				vX.push_back((float)co);
				vY.push_back((float)ro);
			}
		}
	}

	if(vX.size() == 0)
	{
		return false;
	}

	float mean_x = Mean(vX, 0.0f);
	float mean_y = Mean(vY, 0.0f);
	float stdDev_x = StdDev(vX, 0.0f);
	float stdDev_y = StdDev(vY, 0.0f);

	if( (stdDev_x == 0.0) || (stdDev_y == 0.0) )
	{
		return false;
	}

	boundingBox.x = Round(mean_x - sigmaMult * stdDev_x);
	boundingBox.y = Round(mean_y - sigmaMult * stdDev_y);
	boundingBox.width = Round(sigmaMult * stdDev_x * 2.0);
	boundingBox.height = Round(sigmaMult * stdDev_y * 2.0);

	return true;
}


cv::Rect HyperRGBD::Expand(cv::Rect &rect, const float factor, const int maxRight, const int maxBottom)
{
	cv::Rect expanded;
	int widthExp = int( rect.width * factor - rect.width ); 
	int heightExp = int( rect.height * factor - rect.height );
	int commonExp = max(widthExp, heightExp);

	expanded = rect;

	expanded.x -= commonExp/2;
	expanded.y -= commonExp/2;
	expanded.width += commonExp;
	expanded.height += commonExp;

	expanded.x = max(expanded.x, 0);
	expanded.y = max(expanded.y, 0);

	expanded.width = min(expanded.width, maxRight - expanded.x);
	expanded.height = min(expanded.height, maxBottom - expanded.y);

	return expanded;
}




HyperRGBD::StorageMode HyperRGBD::GetStorageModeByExtension(const std::string &absFilename, const StorageMode &storageMode)
{
	StorageMode storageMode_actual = storageMode;
	if(storageMode == STORAGEMODE_EXT_BASED)
	{
		string extension = GetExtension(absFilename);
		if( (extension == "yaml") || (extension == "xml") )
		{
			storageMode_actual = STORAGEMODE_ASCII;
		}
		else if( (extension == "gz") && (GetExtension(RemoveExt(absFilename)) == "xml") )
		{
			storageMode_actual = STORAGEMODE_GZIP;
		}
		else
		{
			throw runtime_error("ERROR (HyperRGBD::Read_Mat) Impossible to infer storageMode from file extension" + extension);
		}
	}

	return storageMode_actual;
}

string HyperRGBD::GetExtension(const string &filename)
{
	int dotPos = (int)filename.find_last_of(".");
	return filename.substr(dotPos+1);
}


string HyperRGBD::RemoveExt(const string &filename)
{
	std::string strTemp = filename;
	unsigned int dotPos = (int)strTemp.find_last_of(".");
	if(dotPos != std::string::npos)
	{
		strTemp.erase(dotPos);
	}
	return strTemp;
}

void HyperRGBD::Write_Mat(const std::string &absFilename, const cv::Mat &mat, const StorageMode &storageMode, const std::string &matName)
{
	StorageMode storageMode_actual = GetStorageModeByExtension(absFilename, storageMode);


	if( (storageMode_actual == STORAGEMODE_ASCII) || (storageMode_actual == STORAGEMODE_GZIP) )
	{
		string absFilename_actual = absFilename;
		if( (storageMode_actual == STORAGEMODE_GZIP) && (GetExtension(absFilename) == "xml") ) absFilename_actual += ".gz";
		if( (storageMode_actual == STORAGEMODE_GZIP) && ( (GetExtension(absFilename) != "gz") || (GetExtension(RemoveExt(absFilename)) != "xml") ) ) absFilename_actual += ".xml.gz";
		if( (storageMode_actual == STORAGEMODE_ASCII) && ( (GetExtension(absFilename) != "yaml") && (GetExtension(absFilename) != "xml") ) ) throw runtime_error("ERROR (MiUt::Write_Mat): wrong extension: " + absFilename);

		FileStorage file( absFilename_actual, FileStorage::WRITE );

		if(!file.isOpened())
		{
			cout << "ERROR (Write_Mat): file not open " << absFilename;
			throw;
		}

		file << matName << mat;

	}
	else if( (storageMode_actual == STORAGEMODE_BINARY) || (storageMode_actual == STORAGEMODE_SNAPPY) )
	{
		std::ofstream oFile(absFilename, ios::binary|ios::out);

		if (!oFile.is_open())
		{
			cout << "ERROR (Write_Mat): file not open " << absFilename;
			throw;
		}

		if(storageMode_actual == STORAGEMODE_BINARY)
		{
			Write_Mat(oFile, mat);
		}
		else
		{
			//serialize
			ostringstream uncompressed;
			Write_Mat(uncompressed, mat);

			//compress
			vector<char> vCompressed;
			Snappy_Compress(uncompressed, vCompressed);

			//write compressed
			Write_Vector(oFile, vCompressed);
		}

	}
	else
	{
		throw runtime_error("ERROR (HyperRGBD::Write_Mat) storageMode_actual");
	}

}


void HyperRGBD::Write_Mat(std::ostream &oFile, const cv::Mat &mat)
{
	oFile.write((char*)&mat.rows, sizeof(mat.rows) );
	oFile.write((char*)&mat.cols, sizeof(mat.cols) );
	int matType = mat.type();
	oFile.write((char*)&matType, sizeof(matType) );
	
	if(mat.isSubmatrix())
	{
		oFile.write((char*)mat.ptr(), mat.step*mat.rows );
	}
	else
	{
		//because it is cropped from another mat
		Mat mat_copy;
		mat.copyTo(mat_copy);
		oFile.write((char*)mat_copy.ptr(), mat_copy.step*mat_copy.rows );
	}
}



void HyperRGBD::Snappy_Uncompress(const char* compressed, const int nCompressedBytes, std::istringstream &uncompressed)
{
#ifndef SNAPPY_IS_NOT_INCLUDED
	size_t nBytes_u;
	snappy_status status = snappy_uncompressed_length(compressed, nCompressedBytes, &nBytes_u);
	if(status != SNAPPY_OK)
	{
		throw runtime_error("ERROR (Snappy_Uncompress): " + (status == SNAPPY_INVALID_INPUT)?"SNAPPY_INVALID_INPUT":"SNAPPY_BUFFER_TOO_SMALL");
	}
	string strUncompressed;
	strUncompressed.resize(nBytes_u);
	status = snappy_uncompress( compressed, nCompressedBytes, (char*)&strUncompressed[0], &nBytes_u);
	if(status != SNAPPY_OK)
	{
		throw runtime_error("ERROR (Snappy_Uncompress): " + (status == SNAPPY_INVALID_INPUT)?"SNAPPY_INVALID_INPUT":"SNAPPY_BUFFER_TOO_SMALL");
	}
	strUncompressed.resize(nBytes_u);
	uncompressed.str(strUncompressed);
#else
	throw runtime_error("ERROR: Include snappy");
#endif
}

void HyperRGBD::Snappy_Uncompress(const std::vector<char> &vCompressed, std::istringstream &uncompressed)
{
	Snappy_Uncompress(&vCompressed[0], (int)vCompressed.size(), uncompressed);
}

void HyperRGBD::Snappy_Compress(std::ostringstream &uncompressed, std::vector<char> &vCompressed)
{
#ifndef SNAPPY_IS_NOT_INCLUDED
	size_t nBytes_c = snappy_max_compressed_length(uncompressed.str().size());
	vCompressed.resize(nBytes_c);
	snappy_status status = snappy_compress(uncompressed.str().c_str(), uncompressed.str().size(), &vCompressed[0], &nBytes_c);
	if(status != SNAPPY_OK)
	{
		throw runtime_error("ERROR (Snappy_Compress): " + (status == SNAPPY_INVALID_INPUT)?"SNAPPY_INVALID_INPUT":"SNAPPY_BUFFER_TOO_SMALL");
	}
	vCompressed.resize(nBytes_c);
#else
	throw runtime_error("ERROR: Include snappy");
#endif
}



void HyperRGBD::Read_Mat(std::istream &iFile, cv::Mat &mat)
{
	int nRows;
	iFile.read((char*)&nRows, sizeof(nRows) );

	int nCols;
	iFile.read((char*)&nCols, sizeof(nCols) );

	int matType;
	iFile.read((char*)&matType, sizeof(matType) );

	mat = Mat(nRows, nCols, matType);
	
	if( (nRows>0) && (nCols>0) )
	{
		iFile.read((char*)mat.ptr(), mat.step*mat.rows );
		if(!iFile.good())
		{
			throw runtime_error("ERROR (MiUt::Read_Mat): (!iFile.good())");
		}
	}
}



void HyperRGBD::Read_Mat(const std::string &absFilename, cv::Mat &mat, const StorageMode &storageMode, const std::string &matName)
{
	StorageMode storageMode_actual = GetStorageModeByExtension(absFilename, storageMode);


	if( (storageMode_actual == STORAGEMODE_ASCII) || (storageMode_actual == STORAGEMODE_GZIP) )
	{
		string absFilename_actual = absFilename;
		if( (storageMode_actual == STORAGEMODE_GZIP) && (GetExtension(absFilename) == "xml") ) absFilename_actual += ".gz";
		if( (storageMode_actual == STORAGEMODE_GZIP) && ( (GetExtension(absFilename) != "gz") || (GetExtension(RemoveExt(absFilename)) != "xml") ) ) absFilename_actual += ".xml.gz";
		if( (storageMode_actual == STORAGEMODE_ASCII) && ( (GetExtension(absFilename) != "yaml") && (GetExtension(absFilename) != "xml") ) ) throw runtime_error("ERROR (MiUt::Read_Mat): wrong extension: " + absFilename);

		FileStorage file(absFilename_actual, FileStorage::READ);

		if(!file.isOpened())
		{
			cout << "ERROR (Read_Mat): file not open " << absFilename;
			throw;
		}

		file[matName] >> mat;

		file.release();

	}
	else if( (storageMode_actual == STORAGEMODE_BINARY) || (storageMode_actual == STORAGEMODE_SNAPPY) )
	{
		std::ifstream iFile(absFilename, ios::binary|ios::out);

		if (!iFile.is_open())
		{
			cout << "ERROR (Read_Mat): file not open " << absFilename;
			throw;
		}

		if(storageMode_actual == STORAGEMODE_BINARY)
		{
			Read_Mat(iFile, mat);
		}
		else
		{
			//read compressed
			vector<char> vCompressed;
			Read_Vector(iFile, vCompressed);

			//uncompress
			istringstream uncompressed;
			Snappy_Uncompress(vCompressed, uncompressed);

			//deserialize
			Read_Mat(uncompressed, mat);
		}

	}
	else
	{
		throw runtime_error("ERROR (MiUt::Read_Mat) storageMode_actual");
	}
}



bool HyperRGBD::FindFilesEndWith(const std::string & path, const std::string & endingString, std::vector<std::string> & foundFiles, bool getOnlyFileName, const int nMaxItems )
{
	foundFiles.clear();

#ifdef _MSC_VER
	// Required structs for searching for files and directories
	WIN32_FIND_DATA FindFileData;
	HANDLE hFind = INVALID_HANDLE_VALUE;

	// Build the file search string...
	char searchDir[2048] = {0};
	char fullpath[2048] = {0};

	// ...if it already is a path that ends with \ or /, add '*'...
	if(path.at(path.length() - 1) == '\\' || path.at(path.length() - 1) == '/')
	{
		_snprintf(searchDir, 2047, "%s*", path.c_str());
		_snprintf(fullpath, 2047, "%s", path.c_str()); // just copy path
	}
	// ...otherwise, add '\*' to the end of the path.
	else
	{
		_snprintf(searchDir, 2047, "%s/*", path.c_str());
		_snprintf(fullpath, 2047, "%s/", path.c_str()); // copy path and add slash (required when building filenames)
	}

	// Find the first file in the directory.
	hFind = FindFirstFile(searchDir, &FindFileData);

	// If there is no file, return
	if (hFind == INVALID_HANDLE_VALUE)
	{
		return false;
	}

	// loop
	do
	{
		// Skip ".", ".." and all directories
		if( ((FindFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0) || strcmp(FindFileData.cFileName, ".") == 0 || strcmp(FindFileData.cFileName, "..") == 0)
		{
			continue;
		}

		if(endingString.size() > std::string(FindFileData.cFileName).size())
		{
			continue;
		}

		// Store filename into the vector if it starts with the given string
		if(endingString.size() > 0)
		{
			if(std::string(FindFileData.cFileName).rfind(endingString) == (std::string(FindFileData.cFileName).size() - endingString.size()) )
			{
				if(getOnlyFileName)
				{
					foundFiles.push_back(FindFileData.cFileName);
				}
				else
				{
					// File found: create a path to the file
					char filePath[2048] = {0};
					_snprintf(filePath, 2047, "%s%s", fullpath, FindFileData.cFileName);
					// Add it to vector of files found
					foundFiles.push_back(filePath);
				}
			}
		}
		else // Always store filename if a starting string has not been provided
		{
			if(getOnlyFileName)
			{
				foundFiles.push_back(FindFileData.cFileName);
			}
			else
			{
				// File found: create a path to the file
				char filePath[2048] = {0};
				_snprintf(filePath, 2047, "%s%s", fullpath, FindFileData.cFileName);
				// Add it to vector of files found
				foundFiles.push_back(filePath);
			}
		}
	}
	// Loop while we find more files
	while(FindNextFile(hFind, &FindFileData) != 0);

	// Release
	FindClose(hFind);

	sort(foundFiles.begin(), foundFiles.end(), StringCompare_Smart_Incr());

	foundFiles.resize( Min(nMaxItems, (int)foundFiles.size() ) );

	return true;
#else // not _MSC_VER
    DIR* directory = opendir(path.c_str());
    if(directory)
    {
        string parent(path);
        if(parent[parent.length()-1] != '/')
            parent.append("/");

        struct dirent dirEntry;
        struct dirent* res = &dirEntry;
        while((readdir_r(directory, &dirEntry, &res) == 0) && (res)) // thread-safe
            if((dirEntry.d_type == DT_REG) &&
                    (strncmp(dirEntry.d_name+(d_namlen-endingString.size()), endingString.c_str(), endingString.length()) == 0) &&
                    (strcmp(dirEntry.d_name, ".") != 0) &&
                    (strcmp(dirEntry.d_name, "..") != 0))
			if(getOnlyFileName)
			{
				foundFiles.push_back(dirEntry.d_name);
			}
			else
			{
                foundFiles.push_back(parent + dirEntry.d_name);
			}
        closedir(directory);

		sort(foundFiles.begin(), foundFiles.end(), StringCompare_Smart_Incr());

		foundFiles.resize( Min(nMaxItems, (int)foundFiles.size() ) );

        return true;
    }

    return false;
#endif // _MSC_VER
}



bool HyperRGBD::FindFilesStartEndWith(const std::string & path, const std::string & startingString, const std::string & endingString, std::vector<std::string> & foundFiles, bool getOnlyFileName, const int nMaxItems )
{
	foundFiles.clear();

#ifdef _MSC_VER
	// Required structs for searching for files and directories
	WIN32_FIND_DATA FindFileData;
	HANDLE hFind = INVALID_HANDLE_VALUE;

	// Build the file search string...
	char searchDir[2048] = {0};
	char fullpath[2048] = {0};

	// ...if it already is a path that ends with \ or /, add '*'...
	if(path.at(path.length() - 1) == '\\' || path.at(path.length() - 1) == '/')
	{
		_snprintf(searchDir, 2047, "%s*", path.c_str());
		_snprintf(fullpath, 2047, "%s", path.c_str()); // just copy path
	}
	// ...otherwise, add '\*' to the end of the path.
	else
	{
		_snprintf(searchDir, 2047, "%s/*", path.c_str());
		_snprintf(fullpath, 2047, "%s/", path.c_str()); // copy path and add slash (required when building filenames)
	}

	// Find the first file in the directory.
	hFind = FindFirstFile(searchDir, &FindFileData);

	// If there is no file, return
	if (hFind == INVALID_HANDLE_VALUE)
	{
		return false;
	}

	// loop
	do
	{
		// Skip ".", ".." and all directories
		if( ((FindFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0) || strcmp(FindFileData.cFileName, ".") == 0 || strcmp(FindFileData.cFileName, "..") == 0)
		{
			continue;
		}

		if(endingString.size() > std::string(FindFileData.cFileName).size())
		{
			continue;
		}

		// Store filename into the vector if it starts and ends with the given strings
		if(startingString.size() > 0 && endingString.size() > 0)
		{
			if( (std::string(FindFileData.cFileName).find(startingString) == 0) && ( std::string(FindFileData.cFileName).find(endingString) == (std::string(FindFileData.cFileName).length() - endingString.length()) ) )
			{
				if(getOnlyFileName)
				{
					foundFiles.push_back(FindFileData.cFileName);
				}
				else
				{
					// File found: create a path to the file
					char filePath[2048] = {0};
					_snprintf(filePath, 2047, "%s%s", fullpath, FindFileData.cFileName);
					// Add it to vector of files found
					foundFiles.push_back(filePath);
				}
			}
		}
		else if(startingString.size() > 0)
		{
			if( std::string(FindFileData.cFileName).find(startingString) == 0 )
			{
				if(getOnlyFileName)
				{
					foundFiles.push_back(FindFileData.cFileName);
				}
				else
				{
					// File found: create a path to the file
					char filePath[2048] = {0};
					_snprintf(filePath, 2047, "%s%s", fullpath, FindFileData.cFileName);
					// Add it to vector of files found
					foundFiles.push_back(filePath);
				}
			}
		}
		else if(endingString.size() > 0)
		{
			if( std::string(FindFileData.cFileName).find(endingString) == (std::string(FindFileData.cFileName).length() - endingString.length()) )
			{
				if(getOnlyFileName)
				{
					foundFiles.push_back(FindFileData.cFileName);
				}
				else
				{
					// File found: create a path to the file
					char filePath[2048] = {0};
					_snprintf(filePath, 2047, "%s%s", fullpath, FindFileData.cFileName);
					// Add it to vector of files found
					foundFiles.push_back(filePath);
				}
			}
		}
		else // Always store filename if a starting and ending strings has not been provided
		{
			if(getOnlyFileName)
			{
				foundFiles.push_back(FindFileData.cFileName);
			}
			else
			{
				// File found: create a path to the file
				char filePath[2048] = {0};
				_snprintf(filePath, 2047, "%s%s", fullpath, FindFileData.cFileName);
				// Add it to vector of files found
				foundFiles.push_back(filePath);
			}
		}
	}
	// Loop while we find more files
	while(FindNextFile(hFind, &FindFileData) != 0);

	// Release
	FindClose(hFind);

	sort(foundFiles.begin(), foundFiles.end(), StringCompare_Smart_Incr());

	foundFiles.resize( Min(nMaxItems, (int)foundFiles.size() ) );

	return true;
#else // not _MSC_VER
    DIR* directory = opendir(path.c_str());
    if(directory)
    {
        string parent(path);
        if(parent[parent.length()-1] != '/')
            parent.append("/");

        struct dirent dirEntry;
        struct dirent* res = &dirEntry;
        while((readdir_r(directory, &dirEntry, &res) == 0) && (res)) // thread-safe
            if((dirEntry.d_type == DT_REG) &&
                    (strncmp(dirEntry.d_name, startingString.c_str(), startingString.length()) == 0) &&
					(strncmp(dirEntry.d_name, endingString.c_str(), endingString.length()) == 0) &&
                    (strcmp(dirEntry.d_name, ".") != 0) &&
                    (strcmp(dirEntry.d_name, "..") != 0))
			if(getOnlyFileName)
			{
				foundFiles.push_back(dirEntry.d_name);
			}
			else
			{
                foundFiles.push_back(parent + dirEntry.d_name);
			}
        closedir(directory);

		sort(foundFiles.begin(), foundFiles.end(), StringCompare_Smart_Incr());

		foundFiles.resize( Min(nMaxItems, (int)foundFiles.size() ) );

        return true;
    }

    return false;
#endif // _MSC_VER
}

size_t HyperRGBD::Sample_uniform(std::vector<bool> &vData, const int step, const int idxFirst)
{
	size_t nSamples = 0;
	for(size_t sa=idxFirst; sa<vData.size(); sa+=step)
	{
		vData[sa] = true;
		nSamples++;
	}

	return nSamples;
}

void HyperRGBD::PushBackCatVect(std::vector<std::pair<std::string, std::pair<size_t, size_t>>> &inVect, const std::string &cat, const size_t &lowLm, const size_t &hiLm)
{
	inVect.push_back(make_pair(cat, make_pair(lowLm, hiLm)));;
}

size_t HyperRGBD::DS_mapToVect(const std::map<std::string, vector<std::string>> &inMap, vector<pair<std::string, vector<std::string>>> &outVect)
{
	typedef std::map<std::string, vector<std::string>>::const_iterator it_type;
	size_t ca;
	it_type it;
	

	outVect.resize(inMap.size());

	for(it = inMap.cbegin(), ca = 0; it != inMap.cend(); it++, ca++)
	{
		outVect[ca].first = it->first;
		outVect[ca].second = it->second;
	}

	return outVect.size();
}

cv::Mat HyperRGBD::GetTextImage(const std::string &label, const double fontScale, const int borderThickness, const bool whiteOnBlack)
{
	int thickness = 1;

	//create image for label
	int baseline=0;
	Size imageSize = getTextSize(label, cv::FONT_HERSHEY_DUPLEX, fontScale, 1, &baseline);
	
	int height = imageSize.height;

	imageSize.height += baseline + thickness;

	Mat textImage = Mat(imageSize, CV_8UC3);
	if(whiteOnBlack)
	{
		textImage = 0;
		cv::putText(textImage, label, cv::Point(0, height), cv::FONT_HERSHEY_DUPLEX, fontScale, CV_RGB(255,255,255), thickness);
		copyMakeBorder(textImage, textImage, borderThickness, borderThickness, borderThickness, borderThickness, BORDER_CONSTANT, CV_RGB(0,0,0) );
	}
	else
	{
		textImage = CV_RGB(255, 255,255);
		cv::putText(textImage, label, cv::Point(0, height), cv::FONT_HERSHEY_DUPLEX, fontScale, CV_RGB(0,0,0), thickness);
		copyMakeBorder(textImage, textImage, borderThickness, borderThickness, borderThickness, borderThickness, BORDER_CONSTANT, CV_RGB(255,255,255) );
	}
	

	return textImage;
}



cv::Mat HyperRGBD::DepthMap2RangeMap(const cv::Mat &depthMap, const float zScaleFactor, float focal_x, float focal_y, float center_x, float center_y)
{
	vector<float> xyOffset(2, 0.0f);
	return DepthMap2RangeMap(depthMap, xyOffset, zScaleFactor, focal_x, focal_y, center_x, center_y);
}

cv::Mat HyperRGBD::DepthMap2RangeMap(const cv::Mat &depthMap, const std::vector<float> &xyOffset, const float zScaleFactor, float focal_x, float focal_y, float center_x, float center_y)
{
	Mat rangeMap = Mat(depthMap.rows, depthMap.cols, CV_32FC3);

	for(int ro=0; ro<depthMap.rows; ro++)
	{
		for(int co=0; co<depthMap.cols; co++)
		{
			rangeMap.at<Vec3f>(ro, co)[2] = depthMap.at<ushort>(ro, co)/zScaleFactor;
			
			rangeMap.at<Vec3f>(ro, co)[0] = (float)(co+xyOffset[0]-center_x);
			rangeMap.at<Vec3f>(ro, co)[1] = (float)(ro+xyOffset[1]-center_y);

			rangeMap.at<Vec3f>(ro, co)[0] *= (float)(rangeMap.at<Vec3f>(ro, co)[2] * (1/focal_x));
			rangeMap.at<Vec3f>(ro, co)[1] *= (float)(rangeMap.at<Vec3f>(ro, co)[2] * (1/focal_y));
		}
	}

	return rangeMap;
}


size_t HyperRGBD::CountDatasetSamples(const std::vector< std::pair< std::string, std::vector<std::string> > > &vDataSet)
{
	size_t nSamples = 0;
	for(size_t cl=0; cl<vDataSet.size(); cl++)
	{
		nSamples += vDataSet[cl].second.size();
	}

	return nSamples;
}