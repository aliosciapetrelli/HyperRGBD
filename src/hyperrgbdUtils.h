#ifndef HYPERRGBD_UTILS_H
#define HYPERRGBD_UTILS_H

#include "hyperrgbdDefines.h"

//#define SNAPPY_IS_NOT_INCLUDED	//Comment to use snappy library


#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>
#include <stdint.h>
//#include <map>
//#include <set>

#include "opencv2/opencv.hpp"



namespace HyperRGBD
{



	//**************************** STATS *************************************


	template<typename Value, typename Accumulator> inline Value Mean(Value* vValues, int nValues, const Accumulator &init = (Value)0)
	{
		Accumulator mean = init;
		for(int i=0; i<nValues; i++)
		{
			mean += (Accumulator)vValues[i];
		}
		if(nValues > 0)
		{
			mean /= nValues;
		}
		return (Value)mean;
	}

	template<typename Value, typename Accumulator> inline Value Mean(const std::vector<Value> &vValues, const Accumulator &init = (Value)0)
	{
		return Mean<Value, Accumulator>((Value*)(&vValues[0]), (int)vValues.size(), init);
	}


	template<typename Value, typename Accumulator> inline Value StdDev(Value* vValues, int nValues, const Accumulator &init = (Value)0)
	{
		Accumulator strDev = init;
		for(int i=0; i<nValues; i++)
		{
			strDev+=((Accumulator)vValues[i]*(Accumulator)vValues[i]);
		}
		if(nValues)
		{
			strDev/=nValues;
		}
		Value average = Mean<Value, Accumulator>(vValues, nValues, init);
		strDev -= ((Accumulator)average*(Accumulator)average);
		strDev = (Accumulator)sqrt((double)strDev);
		return (Value)strDev;
	}

	template<typename Value, typename Accumulator> inline Value StdDev(const std::vector<Value> &vValues, const Accumulator &init = (Value)0)
	{
		return StdDev<Value, Accumulator>((Value*)(&vValues[0]), (int)vValues.size(), init);
	}


    template <typename T> int Round(T val){ return (int)floor(val + 0.5);};

	/*!
	* \brief Compute minimum.
	*
	* \param a,b instances to compare.
	*/
	template <typename Type> inline Type Min(Type a, Type b) { return (a <= b)? a : b; };
	/*!
	* \brief Compute maximum.
	*
	* \param a,b instances to compare.
	*/
	template <typename Type> inline Type Max(Type a, Type b) { return (a >= b)? a : b; };


    inline int NextRandom(int i){ return std::rand()%i;}

	size_t Sample_uniform(std::vector<bool> &vData, const int step, const int idxFirst = 0);

    size_t Sample_random(std::vector<bool> &vData, const float samplingFactor, const unsigned int seed = std::numeric_limits<unsigned int>::max());

	template<typename T> std::string ToString(const T val)
	{
		return static_cast<ostringstream*>( &(ostringstream() << val) )->str();
	}

    bool ExistsFile(const std::string &filename_abs);

    bool ExistsDir(const std::string &directory);
	void CreateDir(const std::string &directory);
    void CreateFullPath(const std::string &directory);

    bool FindDirsStartWith(const std::string & path, const std::string & startingString, std::vector<std::string> & foundDirs, bool getOnlyFileName = false, const int nMaxItems = std::numeric_limits<int>::max());
    bool FindDirsEndWith(const std::string & path, const std::string & endingString, std::vector<std::string> & foundDirs, bool getOnlyFileName = false, const int nMaxItems = std::numeric_limits<int>::max());
    bool FindFilesStartEndWith(const std::string & path, const std::string & startingString, const std::string & endingString, std::vector<std::string> & foundFiles, bool getOnlyFileName = false, const int nMaxItems = std::numeric_limits<int>::max());
    bool FindFilesEndWith(const std::string & path, const std::string & endingString, std::vector<std::string> & foundFiles, bool getOnlyFileName = false, const int nMaxItems = std::numeric_limits<int>::max());

	void Prefix_Path(std::string &filename, const std::string &pathToPrefix);
	void Prefix_Path(std::vector< std::string > &vFilenames, const std::string &pathToPrefix);

    std::string GetPathDir(const std::string &filename);

    std::string GetRelativeName(const std::string &filename);


    std::string GetExtension(const std::string &filename);
	std::string RemoveExt(const std::string &filename);


	template<typename T> T LexicalCast(const std::string& s)
	{
		std::stringstream ss(s);

		T result;
		if ((ss >> result).fail() || !(ss >> std::ws).eof())
		{
			//throw std::bad_cast();
			cout << "ERROR:Impossible to cast " << s;
			getchar();
			exit(-1);
		}

		return result;
	}

	//logical sorting (e.g. Windows explorer)
	class StringCompare_Smart_Incr
	{
	public:
		inline bool operator() (const std::string& a, const std::string& b) const
		{
			unsigned posStr = 0;
			while( (posStr < a.size() ) && (posStr < b.size()) )
			{
				unsigned tkn_idx_a = (unsigned int) a.find_first_of("0123456789", posStr);
				unsigned tkn_idx_b = (unsigned int) b.find_first_of("0123456789", posStr);
				std::string suba = a.substr(posStr, tkn_idx_a - posStr );
				std::string subb = b.substr(posStr, tkn_idx_b - posStr );
				if(suba == subb)
				{
					//same substring

					if(tkn_idx_a == a.size())
					{
						//end of a and at least of b
						return true;
					}

					if(tkn_idx_a == a.size())
					{
						//end of b but not of a
						return false;
					}

					unsigned numberEnd_a = (unsigned int) a.find_first_not_of("0123456789", tkn_idx_a+1);
					unsigned numberEnd_b = (unsigned int) b.find_first_not_of("0123456789", tkn_idx_b+1);
					//check number
					long long number_a = LexicalCast<long long>(a.substr(tkn_idx_a, numberEnd_a - tkn_idx_a));
					long long number_b = LexicalCast<long long>(b.substr(tkn_idx_b, numberEnd_b - tkn_idx_b));
					//long number_a = std::atol(a.substr(tkn_idx_a).c_str());
					//long number_b = std::atol(b.substr(tkn_idx_b).c_str());
					if(number_a != number_b)
					{
						return (number_a < number_b);
					}
				}
				else
				{
					//different substring
					return (suba < subb);
				}
				posStr = (unsigned int) a.find_first_not_of("0123456789", tkn_idx_a + 1);
			}

			return ( a.size() < b.size() );
		}
	};



    
    cv::Rect Expand(cv::Rect &rect, const float factor, const int maxRight = std::numeric_limits<int>::max(), const int maxBottom = std::numeric_limits<int>::max());

    bool FindMaskBoundingBox(cv::Mat &mat, cv::Rect &boundingBox, const float sigmaMult = 2.0f, const unsigned char maskValue = 255);



    enum StorageMode {STORAGEMODE_EXT_BASED, STORAGEMODE_ASCII, STORAGEMODE_BINARY, STORAGEMODE_SNAPPY, STORAGEMODE_GZIP	};

    StorageMode GetStorageModeByExtension(const std::string &absFilename, const StorageMode &storageMode);

    void Read_Mat(std::istream &iFile, cv::Mat &mat); 
    void Write_Mat(std::ostream &oFile, const cv::Mat &mat); 

    void Read_Mat(const std::string &absFilename, cv::Mat &mat, const StorageMode &storageMode = STORAGEMODE_EXT_BASED, const std::string &matName = "Mat"); 
    void Write_Mat(const std::string &absFilename, const cv::Mat &mat, const StorageMode &storageMode = STORAGEMODE_EXT_BASED, const std::string &matName = "Mat"); 


	//************************* Compression **********************************************

	void Snappy_Uncompress(const char* compressed, const int nCompressedBytes, std::istringstream &uncompressed);
	void Snappy_Uncompress(const std::vector<char> &vCompressed, std::istringstream &uncompressed);

	void Snappy_Compress(std::ostringstream &uncompressed, std::vector<char> &vCompressed);



	template <typename T> void Write_Vector(std::ostream &filev, const std::vector<T> &vData, const bool binaryFile = true)
	{
		//if (!filev.is_open() )
		//{
		//	cout << "ERROR (Write_Vector): file is not open";
		//	getchar();
		//	exit(-1);
		//}

		uint64_t nData = (uint64_t)vData.size();
		if(binaryFile)
		{
			//write number of elements
			filev.write((char*)&nData, sizeof(nData) );
			//write data
			filev.write((char*)(&vData[0]), sizeof(T)*nData );
		}
		else
		{
			//write number of elements
			filev << "Vector" << endl;
			filev << nData << endl;
			//write data
			const T* ptrData = &vData[0];
			for(uint64_t da=0; da<nData; da++)
			{
				filev << *ptrData;
				filev << " ";
				ptrData++;
			}
			filev << endl;
		}

		if(!filev.good())
		{
			cout << "ERROR (Write_Vector): !filev.good() [eof fail bad] [" << filev.eof() << " " << filev.fail() << " " << filev.bad() << "]" << endl;
			getchar();
			exit(-1);
		}
	}


    template <typename T> void Read_Vector(std::istream &filev, std::vector<T> &vData, const bool binaryFile = true)
	{
		//if (!filev.is_open() )
		//{
		//	cout << "ERROR (Read_Vector): file is not open";
		//	getchar();
		//	exit(-1);
		//}

		uint64_t nData = 0;
		if(binaryFile)
		{
			//read number of elements
			filev.read((char*)&nData, sizeof(nData) );
			vData.resize((size_t)nData);
			
			if(nData > 0)
			{
				//read data
				filev.read((char*)(&vData[0]), sizeof(T)*nData );
				if(filev.gcount() != (sizeof(T)*nData) )
				{
					cout << "ERROR (Read_Vector): filev.gcount() != (sizeof(T)*nData)  " << filev.gcount() << " " << (sizeof(T)*nData) << endl;
					cout << "Are you sure you opened the file in binary mode?";
					getchar();
					exit(-1);
				}
				if(!filev.good())
				{
					cout << "ERROR (Read_Vector): !filev.good() [eof fail bad] [" << filev.eof() << " " << filev.fail() << " " << filev.bad() << "]" << endl;
					cout << "Are you sure you opened the file in binary mode?";
					getchar();
					exit(-1);
				}
			}
		}
		else
		{
			//read number of elements
			std::string line;
			std::getline(filev, line);
			filev >> nData;
			vData.resize((size_t)nData);

			if(nData > 0)
			{
				//read data
				T* ptrData = &vData[0];
				for(uint64_t da=0; da<nData; da++)
				{
					filev >> (*ptrData);
					ptrData++;
				}
				std::getline(filev, line);

				if(!filev.good())
				{
					cout << "ERROR (Read_Vector): !filev.good() [eof fail bad] [" << filev.eof() << " " << filev.fail() << " " << filev.bad() << "]" << endl;
					getchar();
					exit(-1);
				}
			}
		}
	}


	void PushBackCatVect(std::vector<std::pair<std::string, std::pair<size_t, size_t>>> &inVect, const std::string &cat, const size_t &lowLm, const size_t &hiLm);
	size_t DS_mapToVect(const std::map< std::string, std::vector<std::string> > &inMap, std::vector< std::pair< std::string, std::vector< std::string> > > &outVect);

	cv::Mat GetTextImage(const std::string &label, const double fontScale = 1.0, const int borderThickness = 1, const bool whiteOnBlack = true);

	cv::Mat DepthMap2RangeMap(const cv::Mat &depthMap, const float zScaleFactor=1.0f, float focal_x=570.3f, float focal_y=570.3f, float center_x=240.0f, float center_y=320.0f);
	cv::Mat DepthMap2RangeMap(const cv::Mat &depthMap, const std::vector<float> &xyOffset, const float zScaleFactor=1.0f, float focal_x=570.3f, float focal_y=570.3f, float center_x=240.0f, float center_y=320.0f);

	size_t CountDatasetSamples(const std::vector< std::pair< std::string, std::vector<std::string> > > &vDataSet);
}

#endif