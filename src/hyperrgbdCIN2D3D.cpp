#include "hyperrgbdCIN2D3D.h"
#include "hyperrgbdUtils.h"


using namespace cv;
using namespace std;

void HyperRGBD::CIN2D3D::Init()
{
	m_nMaxCategories_train = numeric_limits<int>::max();
	m_nMaxInstances_train = numeric_limits<int>::max();
	m_nMaxInstanceViews_train = numeric_limits<int>::max();

	m_nMaxCategories_test = numeric_limits<int>::max();
	m_nMaxInstances_test = numeric_limits<int>::max();
	m_nMaxInstanceViews_test = numeric_limits<int>::max();

	m_trial = -1;

	m_evalType = "";

	m_testSetRatio = 1/float(GetNumTrials());

}

HyperRGBD::CIN2D3D::CIN2D3D(const string &absDatasetRoot)
{
	Init();

	SetRoot(absDatasetRoot);
}

HyperRGBD::CIN2D3D::CIN2D3D()
{
	Init();
}



size_t HyperRGBD::CIN2D3D::getDataSet_InstRec(const int trial, vector< pair< string, vector<string> > > &vDataSet, const bool getAbsNames, const bool isTrainingSet, const int nMaxCategories, const int nMaxInstances, const int nMaxInstanceViews)
{
	srand(0);

	size_t nSamples_tot = 0;
	vDataSet.clear();

	string suffix = "_color_8UC3.png";
	size_t suffixSize = suffix.size();

	int nSilverwares = 0;

	//find all category dirs
	
	vector<string> vCatDirs_rel;
	FindDirsEndWith(m_absDatasetRoot, "", vCatDirs_rel, true, nMaxCategories);

	for(size_t ca=0; ca<vCatDirs_rel.size(); ca++)
	{
		cout << "Scan and collect filenames. Cat: " << ca << "/" << vCatDirs_rel.size() << "\r";
		
		if( (vCatDirs_rel[ca] == "Perforator") || (vCatDirs_rel[ca] == "Phone") )
		{
			continue;
		}

		//find all instance dirs in cat dirs
		string catDir_abs = m_absDatasetRoot + "/" + vCatDirs_rel[ca];

		vector<string> vInstanceDirs_rel;
		FindDirsStartWith(catDir_abs, "", vInstanceDirs_rel, true, nMaxInstances);

		//collect all files in all instance dirs
		for(size_t in=0; in<vInstanceDirs_rel.size(); in++)
		{
			vector<string> vCatSamples_all;

			string instanceDir_abs = catDir_abs + "/" + vInstanceDirs_rel[in];

			vector<string> vAllFiles;
			FindFilesStartEndWith(instanceDir_abs, "", suffix, vAllFiles, !getAbsNames, nMaxInstanceViews);

			vector<bool> vTestSampleIds(vAllFiles.size(), false);
			size_t nTestSamples = Sample_uniform(vTestSampleIds, Round(1/m_testSetRatio), trial);
			if(nTestSamples == 0)
			{
				vTestSampleIds[ rand() % vTestSampleIds.size()  ] = true;
			}
			
			vector<string> vSamples;
			for(size_t sa=0; sa<vTestSampleIds.size(); sa++)
			{
				if( (isTrainingSet && !vTestSampleIds[sa]) || (!isTrainingSet && vTestSampleIds[sa]) )
				{
					vAllFiles[sa].resize(vAllFiles[sa].size() - suffixSize);
					vSamples.push_back(vAllFiles[sa]);
				}
			}

			//if(nMaxInstanceViews != numeric_limits<int>::max())
			//{
			//	vSamples.resize(min((size_t)nMaxInstanceViews, vSamples.size()));
			//}

			//Prefix instance Dir
			if(!getAbsNames)
			{
				Prefix_Path(vSamples, vInstanceDirs_rel[in]);
				Prefix_Path(vSamples, vCatDirs_rel[ca]);
			}


			stringstream label;
			//if( (vCatDirs_rel[ca] == "Fork") || (vCatDirs_rel[ca] == "Knife") || (vCatDirs_rel[ca] == "Spoon") )
			//{
			//	label << "Silverware_" << setfill('0') << setw(3) << nSilverwares;
			//	nSilverwares++;
			//}
			//else
			//{
				label << vInstanceDirs_rel[in];
			//}
			vDataSet.push_back( pair< string, vector<string> > (label.str(), vSamples) );
			nSamples_tot += vSamples.size();
		}

	}

	cout << endl;

	return nSamples_tot;

}


size_t HyperRGBD::CIN2D3D::getDataSet_CatRec(const int trial, vector< pair< string, vector<string> > > &vDataSet, const bool getAbsNames, const bool isTrainingSet, const int nMaxCategories, const int nMaxInstances, const int nMaxInstanceViews)
{
	srand(0);
	
	bool isSpecialCatRec = false;
	if(m_evalType == "CatRec_Only1") {isSpecialCatRec = true;}

	size_t nSamples_tot = 0;
	vDataSet.clear();
	m_vDataSet_InstBegins.clear();

	string suffix = "_color_8UC3.png";
	size_t suffixSize = suffix.size();

	//find all category dirs
	vector<string> vCatDirs_rel;
	FindDirsEndWith(m_absDatasetRoot, "", vCatDirs_rel, true, nMaxCategories);

	vector<string> vSamples_silverware;

	for(size_t ca=0; ca<vCatDirs_rel.size(); ca++)
	{
		cout << "Scan and collect filenames. Cat: " << ca << "/" << vCatDirs_rel.size() << "\r";
		
		if( (vCatDirs_rel[ca] == "Perforator") || (vCatDirs_rel[ca] == "Phone") )
		{
			continue;
		}

		//find all instance dirs in cat dirs
		string catDir_abs = m_absDatasetRoot + "/" + vCatDirs_rel[ca];

		vector<string> vInstanceDirs_rel;
		FindDirsStartWith(catDir_abs, "", vInstanceDirs_rel, true, nMaxInstances);

		//select instances 
		vector<bool> vTestInstanceIds(vInstanceDirs_rel.size(), false);
		size_t nTestSamples = 0;

		if(isSpecialCatRec == false)
		{
			nTestSamples = Sample_uniform(vTestInstanceIds, Round(1/m_testSetRatio), trial);
		}

		if(nTestSamples == 0)
		{
			vTestInstanceIds[ rand() % vTestInstanceIds.size() ] = true;
		}

		vector<string> vSamples;

		vector<size_t> vInstBegins;

		for(size_t in=0; in<vTestInstanceIds.size(); in++)
		{
			if( (isTrainingSet && !vTestInstanceIds[in]) || (!isTrainingSet && vTestInstanceIds[in]) )
			{

				string instanceDir_abs = catDir_abs + "/" + vInstanceDirs_rel[in];

				vector<string> vAllFiles;
				FindFilesStartEndWith(instanceDir_abs, "", suffix, vAllFiles, !getAbsNames, nMaxInstanceViews);

				//Prefix instance Dir
				if(!getAbsNames)
				{
					Prefix_Path(vAllFiles, vInstanceDirs_rel[in]);
					Prefix_Path(vAllFiles, vCatDirs_rel[ca]);
				}

				//if(nMaxInstanceViews != numeric_limits<int>::max())
				//{
				//	vAllFiles.resize(min((size_t)nMaxInstanceViews, vAllFiles.size()));
				//}

				vInstBegins.push_back( vSamples.size() );

				vSamples.insert(vSamples.end(), vAllFiles.begin(), vAllFiles.end());
			}
		}

		vInstBegins.push_back( vSamples.size() );

		//remove suffix
		for(size_t sa=0; sa<vSamples.size(); sa++)
		{
			vSamples[sa].resize(vSamples[sa].size() - suffixSize);
		}

		string label;
		//if( (vCatDirs_rel[ca] == "Fork") || (vCatDirs_rel[ca] == "Knife") || (vCatDirs_rel[ca] == "Spoon") )
		//{
		//	
		//	vSamples_silverware.insert(vSamples_silverware.end(), vSamples.begin(), vSamples.end());
		//}
		//else
		//{
			label = vCatDirs_rel[ca];
			vDataSet.push_back( pair< string, vector<string> > (label, vSamples) );

			m_vDataSet_InstBegins.push_back( pair< string, vector< size_t > > (vCatDirs_rel[ca], vInstBegins) );

			nSamples_tot += vSamples.size();
		//}

	}

	//if(vSamples_silverware.size()>0)
	//{
	//	vDataSet.push_back( pair< string, vector<string> > ("Silverware", vSamples_silverware) );
	//	nSamples_tot += vSamples_silverware.size();
	//}

	cout << endl;

	return nSamples_tot;

}




size_t HyperRGBD::CIN2D3D::getTrainingSet_CatRec(const int trial, vector< pair< string, vector<string> > > &vTrainingSet, const bool getAbsNames)
{
	return getDataSet_CatRec(trial, vTrainingSet, getAbsNames, true, m_nMaxCategories_train, m_nMaxInstances_train, m_nMaxInstanceViews_train);
}


size_t HyperRGBD::CIN2D3D::getTestSet_CatRec(const int trial, vector< pair< string, vector<string> > > &vTestSet, const bool getAbsNames)
{
	return getDataSet_CatRec(trial, vTestSet, getAbsNames, false, m_nMaxCategories_test, m_nMaxInstances_test, m_nMaxInstanceViews_test);
}





size_t HyperRGBD::CIN2D3D::getTrainingSet_InstRec(const int trial, vector< pair< string, vector<string> > > &vTrainingSet, const bool getAbsNames)
{
	return getDataSet_InstRec(trial, vTrainingSet, getAbsNames, true, m_nMaxCategories_train, m_nMaxInstances_train, m_nMaxInstanceViews_train);
}



size_t HyperRGBD::CIN2D3D::getTestSet_InstRec(const int trial, vector< pair< string, vector<string> > > &vTestSet, const bool getAbsNames)
{
	return getDataSet_InstRec(trial, vTestSet, getAbsNames, false, m_nMaxCategories_test, m_nMaxInstances_test, m_nMaxInstanceViews_test);
}


size_t HyperRGBD::CIN2D3D::GetTrainingSet(const string &evalType, const int trial, vector< pair< string, vector<string> > > &vTrainingSet, const bool getAbsNames)
{
	if(m_absDatasetRoot == "")
	{
		cout << "ERROR (GetTrainingSet): m_absDatasetRoot not set";
	}

	if(evalType == "CatRec" || evalType == "CatRec_1EveryN" || evalType == "CatRec_Only1")
	{
		return getTrainingSet_CatRec(trial, vTrainingSet, getAbsNames);
	}
	else if(evalType == "CatRec_Full")
	{
		return GetTrainingSet_CatRec_Full(vTrainingSet, getAbsNames);
	}
	else if(evalType == "InstRec" || evalType == "InstRec_1EveryN")
	{
		return getTrainingSet_InstRec(trial, vTrainingSet, getAbsNames);
	}
	else if(evalType == "InstRec_Full")
	{
		return GetTrainingSet_InstRec_Full(vTrainingSet, getAbsNames);
	}
	else
	{
		cout << "ERROR: Uncorrect evalType: " << evalType << endl;
		getchar();
		exit(-1);
	}
}

size_t HyperRGBD::CIN2D3D::GetTestSet(const string &evalType, const int trial, vector< pair< string, vector<string> > > &vTestSet, const bool getAbsNames)
{
	if(m_absDatasetRoot == "")
	{
		cout << "ERROR (GetTestSet): m_absDatasetRoot not set";
	}

	if(evalType == "CatRec" || evalType == "CatRec_1EveryN" || evalType == "CatRec_Only1")
	{
		return getTestSet_CatRec(trial, vTestSet, getAbsNames);
	}
	else if(evalType == "InstRec" || evalType == "InstRec_1EveryN")
	{
		return getTestSet_InstRec(trial, vTestSet, getAbsNames);
	}
	else
	{
		cout << "ERROR: Uncorrect evalType: " << evalType << endl;
		getchar();
		exit(-1);
	}
}

string HyperRGBD::CIN2D3D::GetImageFilename_AbsFull(const string &absFilename_withoutSuffix, const ImageType imageType)
{
	switch(imageType)
	{
	case IMAGETYPE_AS_IS:
		return absFilename_withoutSuffix + "_color_8UC3.png"; 
		break;
	case IMAGETYPE_RGB:
		return absFilename_withoutSuffix + "_color_8UC3.png"; 
		break;
	case IMAGETYPE_DEPTHMAP:
		return absFilename_withoutSuffix + "_xyz_16UC3.png";
		break;
	case IMAGETYPE_RANGEMAP:
		throw runtime_error("ERROR (CIN2D3D::GetImageFilename_AbsFull): IMAGETYPE_RANGEMAP does not exist");
		break;
	case IMAGETYPE_MASK:
		throw runtime_error("ERROR (CIN2D3D::GetImageFilename_AbsFull): IMAGETYPE_MASK does not exist");
		break;
	default:
		throw runtime_error("ERROR (CIN2D3D::GetImageFilename_AbsFull): imageType does not exist");
	}

	return "";
}


void HyperRGBD::CIN2D3D::GetImageFilename_AbsFull(std::vector< std::pair< std::string, std::vector<std::string> > > &vDataSet, const ImageType imageType)
{
	for(size_t cl=0; cl<vDataSet.size(); cl++)
	{
		for(size_t im=0; im<vDataSet[cl].second.size(); im++)
		{
			vDataSet[cl].second[im] = CIN2D3D::GetImageFilename_AbsFull(vDataSet[cl].second[im], imageType);
		}
	}
}




size_t HyperRGBD::CIN2D3D::GetAllAbsFilenames(const int nMaxCategories, const int nMaxInstances, const int nMaxInstanceViews, std::vector< std::string > &vAbsFilenames )
{
	vAbsFilenames.clear();

	string suffix = "_color_8UC3.png";
	
	vector<string> vCatDirs_rel;
	FindDirsEndWith(m_absDatasetRoot, "", vCatDirs_rel, true, nMaxCategories);

	for(size_t ca=0; ca<vCatDirs_rel.size(); ca++)
	{
		cout << "Scan and collect filenames. Cat: " << ca << "/" << vCatDirs_rel.size() << "\r";

		//if( (vCatDirs_rel[ca] == "Perforator") && (vCatDirs_rel[ca] == "Phone") )
		//{
		//	continue;
		//}

		//find all instance dirs in cat dirs
		string catDir_abs = m_absDatasetRoot + "/" + vCatDirs_rel[ca];

		vector<string> vInstanceDirs_rel;
		FindDirsStartWith(catDir_abs, "", vInstanceDirs_rel, true, nMaxInstances);

		//collect all files in all instance dirs
		for(size_t in=0; in<vInstanceDirs_rel.size(); in++)
		{
			vector<string> vSamples;
			string instanceDir_abs = catDir_abs + "/" + vInstanceDirs_rel[in];
			FindFilesEndWith(instanceDir_abs, suffix, vSamples, false, nMaxInstanceViews);

			if( (nMaxInstanceViews == std::numeric_limits<int>::max()) && (vSamples.size() != 36) )
			{
				throw runtime_error("ERROR (CIN2D3D::GetAllAbsFilenames): (vSamples.size() != 36) " + instanceDir_abs);
			}

			vAbsFilenames.insert(vAbsFilenames.end(), vSamples.begin(), vSamples.end() );
		}
	}

	//keep only beginning of filenames
	size_t suffixSize = suffix.size();
	for(size_t sa=0; sa<vAbsFilenames.size(); sa++)
	{
		vAbsFilenames[sa].resize(vAbsFilenames[sa].size() - suffixSize);
	}

	cout << endl;
	return vAbsFilenames.size();
}


cv::Mat HyperRGBD::CIN2D3D::ReadDepthMap(const std::string &absFilename)
{
	cv::Mat xyzImage_16U3 = imread(absFilename, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
	
	Mat channels[3];
	split(xyzImage_16U3, channels);


	////double minVal, maxVal;
	////cv::minMaxLoc(channels[2], &minVal, &maxVal);

	//cv::Mat prova;
	//channels[2].copyTo(prova);

	//prova.convertTo(prova, CV_32F);
	//cv::threshold(prova, prova, 759.0, 0.0, THRESH_TOZERO);

	////channels[0]/=10;
	////channels[1]/=10;
	////channels[2]/=10;

	////channels[0].convertTo(channels[0], CV_8U);
	////channels[1].convertTo(channels[1], CV_8U);
	//prova.convertTo(prova, CV_8U);

	//int dilation_size=2;
	// Mat element = getStructuringElement( MORPH_RECT,
 //                                      Size( 2*dilation_size + 1, 2*dilation_size+1 ),
 //                                      Point( dilation_size, dilation_size ) );

	//dilate(prova, prova, element);

	////cv::imshow("channels_0", channels[0]);
	////cv::imshow("channels_1", channels[1]);
	//cv::imshow("prova", prova);

	//cv::waitKey(0);	

	return channels[2];
}

cv::Mat HyperRGBD::CIN2D3D::ReadRangeMap(const std::string &absFilename)
{
	cv::Mat xyzImage_16U3 = imread(absFilename, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
	
	Mat floatMat;
	xyzImage_16U3.convertTo(floatMat, CV_32F);

	return floatMat;
}

cv::Mat HyperRGBD::CIN2D3D::ReadMask(const std::string &absFilename_withoutSuffix)
{
	cv::Mat image = imread(GetImageFilename_AbsFull(absFilename_withoutSuffix, IMAGETYPE_AS_IS), CV_LOAD_IMAGE_GRAYSCALE | CV_LOAD_IMAGE_ANYDEPTH);
	
	threshold(image, image, 0, 255, THRESH_BINARY);

	//cv::imshow("mask", image);

	//cv::waitKey(0);

	return image;
}


cv::Mat HyperRGBD::CIN2D3D::ReadImage(const std::string &absFilename_withoutSuffix, const ImageType imageType)
{
	switch(imageType)
	{
	case IMAGETYPE_AS_IS:
		return imread(GetImageFilename_AbsFull(absFilename_withoutSuffix, IMAGETYPE_AS_IS), CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH); 
		break;
	case IMAGETYPE_RGB:
		return imread(GetImageFilename_AbsFull(absFilename_withoutSuffix, IMAGETYPE_RGB), CV_LOAD_IMAGE_COLOR | CV_LOAD_IMAGE_ANYDEPTH); 
		break;
	case IMAGETYPE_GREY:
		return imread(GetImageFilename_AbsFull(absFilename_withoutSuffix, IMAGETYPE_GREY), CV_LOAD_IMAGE_GRAYSCALE | CV_LOAD_IMAGE_ANYDEPTH); 
		break;
	case IMAGETYPE_DEPTHMAP:
		return ReadDepthMap(GetImageFilename_AbsFull(absFilename_withoutSuffix, IMAGETYPE_DEPTHMAP)); 
		break;
	case IMAGETYPE_RANGEMAP:
		return ReadRangeMap(GetImageFilename_AbsFull(absFilename_withoutSuffix, IMAGETYPE_DEPTHMAP));
		break;
	case IMAGETYPE_MASK:
		return ReadMask(absFilename_withoutSuffix);
		break;
	case IMAGETYPE_POSE:
		throw runtime_error("ERROR (RGBDObjectDataset::ReadImage): IMAGETYPE_POSE does not exist for Bo dataset");
		break;
	default:
		throw runtime_error("ERROR (RGBDObjectDataset::ReadImage): imageType does not exist");
	}

	return Mat();
}

size_t HyperRGBD::CIN2D3D::GetTrainingSet(std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames)
{
	return GetTrainingSet(m_evalType, m_trial, vTrainingSet, getAbsNames); 
}

size_t HyperRGBD::CIN2D3D::GetTestSet(std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames)
{
	return GetTestSet(m_evalType, m_trial, vTrainingSet, getAbsNames); 
}

size_t HyperRGBD::CIN2D3D::GetAllAbsFilenames(std::vector< std::string > &vAbsFilenames)
{
	const int nMaxCategories = (m_nMaxCategories_test == numeric_limits<int>::max())?m_nMaxCategories_train:max(m_nMaxCategories_train, m_nMaxCategories_test);
	const int nMaxInstances = (m_nMaxInstances_test == numeric_limits<int>::max())?m_nMaxInstances_train:max(m_nMaxInstances_train, m_nMaxInstances_test);
	const int nMaxInstanceViews = (m_nMaxInstanceViews_test == numeric_limits<int>::max())?m_nMaxInstanceViews_train:max(m_nMaxInstanceViews_train, m_nMaxInstanceViews_test);


	return GetAllAbsFilenames(nMaxCategories, nMaxInstances, nMaxInstanceViews, vAbsFilenames );
}

size_t HyperRGBD::CIN2D3D::GetTrainingSet_CatRec_Full(std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames)
{
	//if(m_trial != 0)
	//{
	//	throw runtime_error("ERROR (RGBDObjectDataset::GetTrainingSet_CatRec_Full): (m_trial != 1)   " + ToString(m_trial));
	//}

	size_t nSamples_tot = 0;
	vTrainingSet.clear();
	m_vDataSet_InstBegins.clear();

	//find all category dirs
	vector<string> vCatDirs_rel;
	std::string absDatasetRoot = GetRoot();
	FindDirsEndWith(absDatasetRoot, "", vCatDirs_rel, true, m_nMaxCategories_train);

	

	for(size_t ca=0; ca < vCatDirs_rel.size() ; ca++)
	{

		if( (vCatDirs_rel[ca] == "Perforator") || (vCatDirs_rel[ca] == "Phone") )
		{
			continue;
		}

		cout << "Scan and collect filenames. Cat: " << ca << "/" << vCatDirs_rel.size() << "\r";
		string catDir_abs = absDatasetRoot + "/" + vCatDirs_rel[ca];

		vector<string> vInstanceDirs_rel;
		FindDirsStartWith(catDir_abs, "", vInstanceDirs_rel, true, m_nMaxInstances_train);

		vector<string> vCatSamples_all;
		vector<size_t> vInstBegins(vInstanceDirs_rel.size()+1);

		string suffix = "_color_8UC3.png";

		for(size_t in=0; in < vInstanceDirs_rel.size() ; in++)
		{
			vector<string> vCatSamples;
			string instanceDir_abs = catDir_abs + "/" + vInstanceDirs_rel[in];
			FindFilesEndWith(instanceDir_abs, suffix, vCatSamples, !getAbsNames, m_nMaxInstanceViews_train);

			if(!getAbsNames)
			{
				Prefix_Path(vCatSamples, vInstanceDirs_rel[in]);
				Prefix_Path(vCatSamples, vCatDirs_rel[ca]);
			}

			vInstBegins[in] = vCatSamples_all.size();

			vCatSamples_all.insert(vCatSamples_all.end(), vCatSamples.begin(), vCatSamples.end() );
		}

		vInstBegins[ vInstanceDirs_rel.size() ] = vCatSamples_all.size();

		//keep only beginning of filenames
		size_t suffixSize = suffix.size();
		for(size_t sa=0; sa<vCatSamples_all.size(); sa++)
		{
			vCatSamples_all[sa].resize(vCatSamples_all[sa].size() - suffixSize);
		}



		vTrainingSet.push_back( pair< string, vector<string> > (vCatDirs_rel[ca], vCatSamples_all) );
		m_vDataSet_InstBegins.push_back( pair< string, vector< size_t > > (vCatDirs_rel[ca], vInstBegins) );
		nSamples_tot += vCatSamples_all.size();
	}



	cout << endl;

	return nSamples_tot;
}

size_t HyperRGBD::CIN2D3D::GetTrainingSet_InstRec_Full(std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames)
{
	size_t nSamples_tot = 0;
	vTrainingSet.clear();

	string suffix = "_color_8UC3.png";
	size_t suffixSize = suffix.size();

	int nSilverwares = 0;

	//find all category dirs
	
	vector<string> vCatDirs_rel;
	FindDirsEndWith(m_absDatasetRoot, "", vCatDirs_rel, true, m_nMaxCategories_train);

	for(size_t ca=0; ca<vCatDirs_rel.size(); ca++)
	{
		cout << "Scan and collect filenames. Cat: " << ca << "/" << vCatDirs_rel.size() << "\r";
		
		if( (vCatDirs_rel[ca] == "Perforator") || (vCatDirs_rel[ca] == "Phone") )
		{
			continue;
		}

		//find all instance dirs in cat dirs
		string catDir_abs = m_absDatasetRoot + "/" + vCatDirs_rel[ca];

		vector<string> vInstanceDirs_rel;
		FindDirsStartWith(catDir_abs, "", vInstanceDirs_rel, true, m_nMaxInstances_train);

		//collect all files in all instance dirs
		for(size_t in=0; in<vInstanceDirs_rel.size(); in++)
		{
			vector<string> vCatSamples_all;

			string instanceDir_abs = catDir_abs + "/" + vInstanceDirs_rel[in];

			vector<string> vAllFiles;
			FindFilesStartEndWith(instanceDir_abs, "", suffix, vAllFiles, !getAbsNames, m_nMaxInstanceViews_train);

			
			vector<string> vSamples;
			for(size_t sa=0; sa<vAllFiles.size(); sa++)
			{
				vAllFiles[sa].resize(vAllFiles[sa].size() - suffixSize);
				vSamples.push_back(vAllFiles[sa]);
			}

			//if(nMaxInstanceViews != numeric_limits<int>::max())
			//{
			//	vSamples.resize(min((size_t)nMaxInstanceViews, vSamples.size()));
			//}

			//Prefix instance Dir
			if(!getAbsNames)
			{
				Prefix_Path(vSamples, vInstanceDirs_rel[in]);
				Prefix_Path(vSamples, vCatDirs_rel[ca]);
			}


			stringstream label;
			//if( (vCatDirs_rel[ca] == "Fork") || (vCatDirs_rel[ca] == "Knife") || (vCatDirs_rel[ca] == "Spoon") )
			//{
			//	label << "Silverware_" << setfill('0') << setw(3) << nSilverwares;
			//	nSilverwares++;
			//}
			//else
			//{
			label << vInstanceDirs_rel[in];
			//}
			vTrainingSet.push_back( pair< string, vector<string> > (label.str(), vSamples) );
			nSamples_tot += vSamples.size();
		}

	}

	cout << endl;

	return nSamples_tot;
}

