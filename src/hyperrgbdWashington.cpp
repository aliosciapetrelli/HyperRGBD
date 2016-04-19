#include "hyperrgbdWashington.h"
#include "hyperrgbdUtils.h"


using namespace cv;
using namespace std;



void HyperRGBD::Washington::Init()
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

HyperRGBD::Washington::Washington(const string &absDatasetRoot)
{
	Init();

	SetRoot(absDatasetRoot);
}

HyperRGBD::Washington::Washington()
{
	Init();
}


void HyperRGBD::Washington::SetRoot(const string &absDatasetRoot)
{
	m_absDatasetRoot = absDatasetRoot;

	m_absLeaveOneOutCategoryInstances = m_absDatasetRoot + "/testinstance_ids.txt";

	m_absEvalDir = m_absDatasetRoot + "/rgbd-dataset_eval";
	m_absCropDir = m_absDatasetRoot + "/rgbd-dataset";
}


void HyperRGBD::Washington::ParseLeaveOneOutCategoryInstances(const string &absLeaveOneOutCategoryInstances, vector< set<string> > &vsLeaveOneOutCategoryInstances)
{
	ifstream inFile(absLeaveOneOutCategoryInstances);

	if(!inFile.is_open())
	{
		cout << "ERROR (ParseLeaveOneOutCategoryInstances): impossible to open " << absLeaveOneOutCategoryInstances << endl;
		getchar();
		exit(-1);
	}

	vsLeaveOneOutCategoryInstances.clear();
	vsLeaveOneOutCategoryInstances.resize(GetNumTrials());
	string line;
	for(int tr=0; tr<GetNumTrials(); tr++)
	{
		getline(inFile, line);

		while(true)
		{
			getline(inFile, line);
			if(line == "")
			{
				break;
			}
			vsLeaveOneOutCategoryInstances[tr].insert(line);
		}
		
		getline(inFile, line);
	}

}


size_t HyperRGBD::Washington::GetTrainingSet_Category(const int trial, vector< pair< string, vector<string> > > &vTrainingSet, const bool getAbsNames)
{
	if(trial < 0 || trial >= GetNumTrials())
	{
		cout << "ERROR (GetTrainingSet_Category): (trial < 0 || trial > 9)   " << trial;
		getchar();
		exit(-1);
	}

	size_t nSamples_tot = 0;
	vTrainingSet.clear();
	m_vDataSet_InstBegins.clear();

	//find all category dirs
	
	vector<string> vCatDirs_rel;
	FindDirsEndWith(m_absEvalDir, "", vCatDirs_rel, true, m_nMaxCategories_train);

	m_vDataSet_InstBegins.resize(vCatDirs_rel.size());
	for(size_t ca=0; ca<vCatDirs_rel.size(); ca++)
	{
		m_vDataSet_InstBegins[ca].first = vCatDirs_rel[ca];

		cout << "Scan and collect filenames. Cat: " << ca << "/" << vCatDirs_rel.size() << "\r";
		//find all instance dirs in cat dirs
		string catDir_abs = m_absEvalDir + "/" + vCatDirs_rel[ca];

		vector<string> vInstanceDirs_rel;
		FindDirsStartWith(catDir_abs, "", vInstanceDirs_rel, true, m_nMaxInstances_train);

		//collect all files in all instance dirs
		vector<string> vCatSamples_all;
		string suffix = "_crop.png";
		for(size_t in=0; in<vInstanceDirs_rel.size(); in++)
		{
			//skip instance used in testing (leave one instance out) 
			if(m_vsLeaveOneOutCategoryInstances[trial].find(vInstanceDirs_rel[in]) != m_vsLeaveOneOutCategoryInstances[trial].end() )
			{
				continue;
			}

			vector<string> vCatSamples;
			string instanceDir_abs = catDir_abs + "/" + vInstanceDirs_rel[in];
			FindFilesEndWith(instanceDir_abs, suffix, vCatSamples, !getAbsNames, m_nMaxInstanceViews_train);

			if(!getAbsNames)
			{
				Prefix_Path(vCatSamples, vInstanceDirs_rel[in]);
				Prefix_Path(vCatSamples, vCatDirs_rel[ca]);
			}

			m_vDataSet_InstBegins[ca].second.push_back(vCatSamples_all.size());

			vCatSamples_all.insert(vCatSamples_all.end(), vCatSamples.begin(), vCatSamples.end() );
		}
		m_vDataSet_InstBegins[ca].second.push_back(vCatSamples_all.size());

		//keep only beginning of filenames
		size_t suffixSize = suffix.size();
		for(size_t sa=0; sa<vCatSamples_all.size(); sa++)
		{
			vCatSamples_all[sa].resize(vCatSamples_all[sa].size() - suffixSize);
		}

		vTrainingSet.push_back( pair< string, vector<string> > (vCatDirs_rel[ca], vCatSamples_all) );
		nSamples_tot += vCatSamples_all.size();
	}

	cout << endl;
	return nSamples_tot;
}


size_t HyperRGBD::Washington::GetTestSet_Category(const int trial, vector< pair< string, vector<string> > > &vTestSet, const bool getAbsNames)
{

	if(trial < 0 || trial >= GetNumTrials())
	{
		cout << "ERROR (GetTestSet_Category): (trial < 0 || trial > 9)   " << trial;
		getchar();
		exit(-1);
	}

	size_t nSamples_tot = 0;
	vTestSet.clear();

	//find all category dirs
	vector<string> vCatDirs_rel;
	FindDirsEndWith(m_absEvalDir, "", vCatDirs_rel, true, m_nMaxCategories_test);

	for(size_t ca=0; ca<vCatDirs_rel.size(); ca++)
	{
		cout << "Scan and collect filenames. Cat: " << ca << "/" << vCatDirs_rel.size() << "\r";
		//find all instance dirs in cat dirs
		string catDir_abs = m_absEvalDir + "/" + vCatDirs_rel[ca];

		vector<string> vInstanceDirs_rel;
		FindDirsStartWith(catDir_abs, "", vInstanceDirs_rel, true, m_nMaxInstances_test);

		//collect all files in all instance dirs
		vector<string> vCatSamples_all;
		string suffix = "_crop.png";
		for(size_t in=0; in<vInstanceDirs_rel.size(); in++)
		{
			//skip instance used in training (leave one instance out) 
			if(m_vsLeaveOneOutCategoryInstances[trial].find(vInstanceDirs_rel[in]) == m_vsLeaveOneOutCategoryInstances[trial].end() )
			{
				continue;
			}

			vector<string> vCatSamples;
			string instanceDir_abs = catDir_abs + "/" + vInstanceDirs_rel[in];
			FindFilesEndWith(instanceDir_abs, suffix, vCatSamples, !getAbsNames, m_nMaxInstanceViews_test);

			if(!getAbsNames)
			{
				Prefix_Path(vCatSamples, vInstanceDirs_rel[in]);
				Prefix_Path(vCatSamples, vCatDirs_rel[ca]);
			}

			vCatSamples_all.insert(vCatSamples_all.end(), vCatSamples.begin(), vCatSamples.end() );
		}

		//keep only beginning of filenames
		size_t suffixSize = suffix.size();
		for(size_t sa=0; sa<vCatSamples_all.size(); sa++)
		{
			vCatSamples_all[sa].resize(vCatSamples_all[sa].size() - suffixSize);
		}

		vTestSet.push_back( pair< string, vector<string> > (vCatDirs_rel[ca], vCatSamples_all) );
		nSamples_tot += vCatSamples_all.size();
	}

	cout << endl;

	return nSamples_tot;
}


size_t HyperRGBD::Washington::GetTraining_And_Test_Set_InstanceRec_AlternatingContiguousFrames(const int trial, map< string, std::vector<int> >  &mvLeaveTwoOutInstanceSequences, vector< pair< string, vector<string> > > &vTrainingSet, vector< pair< string, vector<string> > > &vTestSet, const bool getAbsNames)
{

	if(trial < 0 || trial >= GetNumTrials())
	{
		cout << "ERROR (GetTraining_And_Test_Set_InstanceRec_AlternatingContiguousFrames): (trial < 0 || trial > 9)   " << trial;
		getchar();
		exit(-1);
	}

	cout << "GetTraining_And_Test_Set_InstanceRec_AlternatingContiguousFrames" << endl;

	size_t nSamples_tot = 0;
	vTestSet.clear();
	vTrainingSet.clear();

	CvRNG rng = cvRNG(trial);

	string suffix = "_crop.png";
	string prefix = "";

	vector<string> vCatDirs_rel;
	FindDirsEndWith(m_absEvalDir, "", vCatDirs_rel, true, m_nMaxCategories_train);

	for(size_t ca=0; ca<vCatDirs_rel.size(); ca++)
	{
		cout << "Scan and collect filenames. Cat: " << ca << "/" << vCatDirs_rel.size() << "\r";
		//find all instance dirs in cat dirs
		string catDir_abs = m_absEvalDir + "/" + vCatDirs_rel[ca];

		vector<string> vInstanceDirs_rel;
		FindDirsStartWith(catDir_abs, "", vInstanceDirs_rel, true, m_nMaxInstances_train);

		//collect all files in all instance dirs
		

		for(size_t in=0; in<vInstanceDirs_rel.size(); in++)
		{
			vector< vector<string> > vvSamples(9);

			

			string instanceDir_abs = catDir_abs + "/" + vInstanceDirs_rel[in];

			//get 30 degrees files
			vector<string> vSamples;
			prefix = vInstanceDirs_rel[in] + "_1_";
			FindFilesStartEndWith(instanceDir_abs, prefix, suffix, vSamples, !getAbsNames, m_nMaxInstanceViews_train);

			double oneThird = vSamples.size()/3.0;
			vvSamples[0].insert(vvSamples[0].end(), vSamples.begin(), vSamples.begin() + (size_t)oneThird);
			vvSamples[1].insert(vvSamples[1].end(), vSamples.begin() + (size_t)(oneThird), vSamples.begin() + (size_t)(2.0*oneThird));
			vvSamples[2].insert(vvSamples[2].end(), vSamples.begin() + (size_t)(2.0*oneThird), vSamples.end() );

			//get 45 degrees files
			vSamples.clear();
			prefix = vInstanceDirs_rel[in] + "_2_";
			FindFilesStartEndWith(instanceDir_abs, prefix, suffix, vSamples, !getAbsNames, m_nMaxInstanceViews_train);

			oneThird = vSamples.size()/3.0;
			vvSamples[3].insert(vvSamples[3].end(), vSamples.begin(), vSamples.begin() + (size_t)oneThird);
			vvSamples[4].insert(vvSamples[4].end(), vSamples.begin() + (size_t)(oneThird), vSamples.begin() + (size_t)(2.0*oneThird));
			vvSamples[5].insert(vvSamples[5].end(), vSamples.begin() + (size_t)(2.0*oneThird), vSamples.end() );

			//get 60 degrees files
			vSamples.clear();
			prefix = vInstanceDirs_rel[in] + "_4_";
			FindFilesStartEndWith(instanceDir_abs, prefix, suffix, vSamples, !getAbsNames, m_nMaxInstanceViews_train);

			oneThird = vSamples.size()/3.0;
			vvSamples[6].insert(vvSamples[6].end(), vSamples.begin(), vSamples.begin() + (size_t)oneThird);
			vvSamples[7].insert(vvSamples[7].end(), vSamples.begin() + (size_t)(oneThird), vSamples.begin() + (size_t)(2.0*oneThird));
			vvSamples[8].insert(vvSamples[8].end(), vSamples.begin() + (size_t)(2.0*oneThird), vSamples.end() );

			//randomly generate the two left out
			int firstLeftOut = cvRandInt(&rng) % 9;
			int secondLeftOut = 0;
			do
			{
				secondLeftOut = cvRandInt(&rng) % 9;
			}
			while(secondLeftOut == firstLeftOut);

			vector<string> vSamples_training;
			vector<string> vSamples_test;
			for(size_t pa=0; pa<vvSamples.size(); pa++)
			{
				if( (pa == firstLeftOut) || (pa == secondLeftOut) )
				{
					vSamples_test.insert(vSamples_test.end(), vvSamples[pa].begin(), vvSamples[pa].end() );
				}
				else
				{
					vSamples_training.insert(vSamples_training.end(), vvSamples[pa].begin(), vvSamples[pa].end() );
				}
			}

			if(!getAbsNames)
			{
				Prefix_Path(vSamples_test, vInstanceDirs_rel[in]);
				Prefix_Path(vSamples_test, vCatDirs_rel[ca]);

				Prefix_Path(vSamples_training, vInstanceDirs_rel[in]);
				Prefix_Path(vSamples_training, vCatDirs_rel[ca]);
			}


			//keep only beginning of filenames
			size_t suffixSize = suffix.size();
			for(size_t sa=0; sa<vSamples_test.size(); sa++)
			{
				vSamples_test[sa].resize(vSamples_test[sa].size() - suffixSize);
			}
			for(size_t sa=0; sa<vSamples_training.size(); sa++)
			{
				vSamples_training[sa].resize(vSamples_training[sa].size() - suffixSize);
			}

			vTestSet.push_back( pair< string, vector<string> > (vInstanceDirs_rel[in], vSamples_test) );
			vTrainingSet.push_back( pair< string, vector<string> > (vInstanceDirs_rel[in], vSamples_training) );
			nSamples_tot += vSamples_training.size();

		}
	}



	cout << endl;

	return nSamples_tot;
}


size_t HyperRGBD::Washington::GetTrainingSet_InstanceRec_LeaveSequenceOut(vector< pair< string, vector<string> > > &vTrainingSet, const bool getAbsNames)
{
	size_t nSamples_tot = 0;
	vTrainingSet.clear();

	//find all category dirs
	
	vector<string> vCatDirs_rel;
	FindDirsEndWith(m_absEvalDir, "", vCatDirs_rel, true, m_nMaxCategories_train);

	for(size_t ca=0; ca<vCatDirs_rel.size(); ca++)
	{
		cout << "Scan and collect filenames. Cat: " << ca << "/" << vCatDirs_rel.size() << "\r";
		//find all instance dirs in cat dirs
		string catDir_abs = m_absEvalDir + "/" + vCatDirs_rel[ca];

		vector<string> vInstanceDirs_rel;
		FindDirsStartWith(catDir_abs, "", vInstanceDirs_rel, true, m_nMaxInstances_train);

		//collect all files in all instance dirs
		
		string suffix = "_crop.png";
		string prefix = "";
		for(size_t in=0; in<vInstanceDirs_rel.size(); in++)
		{
			vector<string> vCatSamples_all;

			string instanceDir_abs = catDir_abs + "/" + vInstanceDirs_rel[in];

			//get 30 degrees files
			vector<string> vCatSamples_30degress;
			prefix = vInstanceDirs_rel[in] + "_1_";
			FindFilesStartEndWith(instanceDir_abs, prefix, suffix, vCatSamples_30degress, !getAbsNames, m_nMaxInstanceViews_train);
			if(!getAbsNames)
			{
				Prefix_Path(vCatSamples_30degress, vInstanceDirs_rel[in]);
				Prefix_Path(vCatSamples_30degress, vCatDirs_rel[ca]);
			}
			vCatSamples_all.insert(vCatSamples_all.end(), vCatSamples_30degress.begin(), vCatSamples_30degress.end() );

			//get 60 degrees files
			vector<string> vCatSamples_60degress;
			prefix = vInstanceDirs_rel[in] + "_4_";
			FindFilesStartEndWith(instanceDir_abs, prefix, suffix, vCatSamples_60degress, !getAbsNames, m_nMaxInstanceViews_train);
			if(!getAbsNames)
			{
				Prefix_Path(vCatSamples_60degress, vInstanceDirs_rel[in]);
				Prefix_Path(vCatSamples_60degress, vCatDirs_rel[ca]);
			}
			vCatSamples_all.insert(vCatSamples_all.end(), vCatSamples_60degress.begin(), vCatSamples_60degress.end() );

			//keep only beginning of filenames
			size_t suffixSize = suffix.size();
			for(size_t sa=0; sa<vCatSamples_all.size(); sa++)
			{
				vCatSamples_all[sa].resize(vCatSamples_all[sa].size() - suffixSize);
			}

			vTrainingSet.push_back( pair< string, vector<string> > (vInstanceDirs_rel[in], vCatSamples_all) );
			nSamples_tot += vCatSamples_all.size();

		}
	}

	cout << endl;

	return nSamples_tot;

}



size_t HyperRGBD::Washington::GetTestSet_InstanceRec_LeaveSequenceOut(vector< pair< string, vector<string> > > &vTestSet, const bool getAbsNames)
{
	size_t nSamples_tot = 0;
	vTestSet.clear();

	//find all category dirs
	
	vector<string> vCatDirs_rel;
	FindDirsEndWith(m_absEvalDir, "", vCatDirs_rel, true, m_nMaxCategories_test);

	for(size_t ca=0; ca<vCatDirs_rel.size(); ca++)
	{
		cout << "Scan and collect filenames. Cat: " << ca << "/" << vCatDirs_rel.size() << "\r";
		//find all instance dirs in cat dirs
		string catDir_abs = m_absEvalDir + "/" + vCatDirs_rel[ca];

		vector<string> vInstanceDirs_rel;
		FindDirsStartWith(catDir_abs, "", vInstanceDirs_rel, true, m_nMaxInstances_test);

		//collect all files in all instance dirs
		string suffix = "_crop.png";
		string prefix = "";
		for(size_t in=0; in<vInstanceDirs_rel.size(); in++)
		{
			vector<string> vCatSamples_all;

			string instanceDir_abs = catDir_abs + "/" + vInstanceDirs_rel[in];

			//get 45 degrees files
			vector<string> vCatSamples_45degress;
			prefix = vInstanceDirs_rel[in] + "_2_";
			FindFilesStartEndWith(instanceDir_abs, prefix, suffix, vCatSamples_45degress, !getAbsNames, m_nMaxInstanceViews_test);
			if(!getAbsNames)
			{
				Prefix_Path(vCatSamples_45degress, vInstanceDirs_rel[in]);
				Prefix_Path(vCatSamples_45degress, vCatDirs_rel[ca]);
			}
			vCatSamples_all.insert(vCatSamples_all.end(), vCatSamples_45degress.begin(), vCatSamples_45degress.end() );
		

			//keep only beginning of filenames
			size_t suffixSize = suffix.size();
			for(size_t sa=0; sa<vCatSamples_all.size(); sa++)
			{
				vCatSamples_all[sa].resize(vCatSamples_all[sa].size() - suffixSize);
			}

			vTestSet.push_back( pair< string, vector<string> > (vInstanceDirs_rel[in], vCatSamples_all) );
			nSamples_tot += vCatSamples_all.size();
		}
	}

	cout << endl;

	return nSamples_tot;

}

size_t HyperRGBD::Washington::GetTrainingSet(const string &evalType, vector< pair< string, vector<string> > > &vTrainingSet, const bool getAbsNames)
{
	if(m_absDatasetRoot == "")
	{
		cout << "ERROR (GetTrainingSet): m_absDatasetRoot not set";
	}

	if(evalType == "CatRec")
	{
		ParseLeaveOneOutCategoryInstances(m_absLeaveOneOutCategoryInstances, m_vsLeaveOneOutCategoryInstances);

		return GetTrainingSet_Category(m_trial, vTrainingSet, getAbsNames);
	}
	else if(evalType == "InstRec_ACF")
	{
		return GetTraining_And_Test_Set_InstanceRec_AlternatingContiguousFrames(m_trial, m_mvLeaveTwoOutInstanceSequences, vTrainingSet, m_vTestSet, getAbsNames);
	}
	else if(evalType == "InstRec_LSO")
	{
		return GetTrainingSet_InstanceRec_LeaveSequenceOut(vTrainingSet, getAbsNames);
	}
	else if(evalType == "CatRec_Full")
	{
		return GetTrainingSet_CatRec_Full(vTrainingSet, getAbsNames);
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


size_t HyperRGBD::Washington::GetTestSet(const string &evalType, vector< pair< string, vector<string> > > &vTestSet, const bool getAbsNames)
{
	if(m_absDatasetRoot == "")
	{
		cout << "ERROR (GetTestSet): m_absDatasetRoot not set";
	}

	if(evalType == "CatRec" )
	{
		ParseLeaveOneOutCategoryInstances(m_absLeaveOneOutCategoryInstances, m_vsLeaveOneOutCategoryInstances);

		return GetTestSet_Category(m_trial, vTestSet, getAbsNames);
	}
	else if(evalType == "InstRec_ACF")
	{
		return GetTraining_And_Test_Set_InstanceRec_AlternatingContiguousFrames(m_trial, m_mvLeaveTwoOutInstanceSequences, m_vTrainingSet, vTestSet, getAbsNames);
	}
	else if(evalType == "InstRec_LSO")
	{
		return GetTestSet_InstanceRec_LeaveSequenceOut(vTestSet, getAbsNames);
	}
	else
	{
		cout << "ERROR: Uncorrect evalType: " << evalType << endl;
		getchar();
		exit(-1);
	}
}


string HyperRGBD::Washington::GetImageFilename_AbsFull(const string &absFilename_withoutSuffix, const ImageType imageType)
{
	switch(imageType)
	{
	case IMAGETYPE_AS_IS:
		return absFilename_withoutSuffix + "_crop.png"; 
		break;
	case IMAGETYPE_RGB:
		return absFilename_withoutSuffix + "_crop.png"; 
		break;
	case IMAGETYPE_DEPTHMAP:
		return absFilename_withoutSuffix + "_depthcrop.png";
		break;
	case IMAGETYPE_RANGEMAP:
		throw runtime_error("ERROR (GetImageFilename_AbsFull): IMAGETYPE_RANGEMAP does not exist");
		break;
	case IMAGETYPE_MASK:
		return absFilename_withoutSuffix + "_maskcrop.png";
		break;
	default:
		throw runtime_error("ERROR (GetImageFilename_AbsFull): imageType does not exist");
	}

	return "";
}


void HyperRGBD::Washington::GetImageFilename_AbsFull(std::vector< std::pair< std::string, std::vector<std::string> > > &vDataSet, const ImageType imageType)
{
	for(size_t cl=0; cl<vDataSet.size(); cl++)
	{
		for(size_t im=0; im<vDataSet[cl].second.size(); im++)
		{
			vDataSet[cl].second[im] = HyperRGBD::Washington::GetImageFilename_AbsFull(vDataSet[cl].second[im], imageType);
		}
	}
}


void HyperRGBD::Washington::ReadTopLeftLoc( const string &filename, vector<float> &topLeft )
{
	std::ifstream instream;
	string ss;
	topLeft.resize(2,1);

	if (!ExistsFile(filename))
	{
		cout << "ERROR (ReadTopLeftLoc): file not found: " << filename << endl;
		getchar();
		exit(-1);
	}

	instream.open( filename.c_str() );
	getline( instream, ss );
	size_t pos=ss.find(",");
	topLeft.resize(2);
	topLeft[0]=(float)atoi( ss.substr(0,pos).c_str() );
	topLeft[1]=(float)atoi( ss.substr(pos+1).c_str() );

	instream.close();

	return;
}



size_t HyperRGBD::Washington::GetAllAbsFilenames(const int nMaxCategories, const int nMaxInstances, const int nMaxInstanceViews, std::vector< std::string > &vAbsFilenames )
{
	vAbsFilenames.clear();

	string suffix = "_crop.png";
	
	vector<string> vCatDirs_rel;
	FindDirsEndWith(m_absEvalDir, "", vCatDirs_rel, true, nMaxCategories);

	for(size_t ca=0; ca<vCatDirs_rel.size(); ca++)
	{
		cout << "Scan and collect filenames. Cat: " << ca << "/" << vCatDirs_rel.size() << "\r";
		//find all instance dirs in cat dirs
		string catDir_abs = m_absEvalDir + "/" + vCatDirs_rel[ca];

		vector<string> vInstanceDirs_rel;
		FindDirsStartWith(catDir_abs, "", vInstanceDirs_rel, true, nMaxInstances);

		//collect all files in all instance dirs
		for(size_t in=0; in<vInstanceDirs_rel.size(); in++)
		{
			vector<string> vSamples;
			string instanceDir_abs = catDir_abs + "/" + vInstanceDirs_rel[in];
			FindFilesEndWith(instanceDir_abs, suffix, vSamples, false, nMaxInstanceViews);

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


cv::Mat HyperRGBD::Washington::ReadImage(const std::string &absFilename_withoutSuffix, const ImageType imageType)
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
		return imread(GetImageFilename_AbsFull(absFilename_withoutSuffix, IMAGETYPE_DEPTHMAP), CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH); 
		break;
	case IMAGETYPE_RANGEMAP:
		return ReadRangeMap(absFilename_withoutSuffix);
		break;
	case IMAGETYPE_MASK:
		return imread(GetImageFilename_AbsFull(absFilename_withoutSuffix, IMAGETYPE_MASK), CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH); 
		break;
	case IMAGETYPE_POSE:
		throw runtime_error("ERROR (ReadImage): IMAGETYPE_POSE does not exist for Bo dataset");
		break;
	default:
		throw runtime_error("ERROR (ReadImage): imageType does not exist");
	}
}

cv::Mat HyperRGBD::Washington::ReadRangeMap(const std::string &absFilename_withoutSuffix)
{
	vector<float> topLeftDispl;
	ReadTopLeftLoc( GetTopLeftLocFilename_AbsFull(absFilename_withoutSuffix), topLeftDispl );

	cv::Mat depthMap = imread(GetImageFilename_AbsFull(absFilename_withoutSuffix, IMAGETYPE_DEPTHMAP), CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
	return DepthMap2RangeMap(depthMap, topLeftDispl);
}



cv::Mat HyperRGBD::Washington::DepthMap2RangeMap(const cv::Mat &depthMap, const std::vector<float> &topLeftDispl, double focal, double ycenter, double xcenter)
{
	cv::Mat rangeMap = Mat(depthMap.rows, depthMap.cols, CV_32FC3);

	for(int ro=0; ro<depthMap.rows; ro++)
	{
		for(int co=0; co<depthMap.cols; co++)
		{
			rangeMap.at<Vec3f>(ro, co)[2] = depthMap.at<ushort>(ro, co);
			
			rangeMap.at<Vec3f>(ro, co)[0] = (float)(co+1);
			rangeMap.at<Vec3f>(ro, co)[1] = (float)(ro+1);

			rangeMap.at<Vec3f>(ro, co)[0] += (float)((topLeftDispl[0]-1)-xcenter);
			rangeMap.at<Vec3f>(ro, co)[1] += (float)((topLeftDispl[1]-1)-ycenter);

			//rangeMap.at<Vec3f>(ro, co)[0] *= (rangeMap.at<Vec3f>(ro, co)[2] * (float)(1/focal));
			//rangeMap.at<Vec3f>(ro, co)[1] *= (rangeMap.at<Vec3f>(ro, co)[2] * (float)(1/focal));
			rangeMap.at<Vec3f>(ro, co)[0] *= (float)(rangeMap.at<Vec3f>(ro, co)[2] * (1/focal));
			rangeMap.at<Vec3f>(ro, co)[1] *= (float)(rangeMap.at<Vec3f>(ro, co)[2] * (1/focal));
		}
	}

	return rangeMap;
}

size_t HyperRGBD::Washington::GetTrainingSet(std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames)
{
	return GetTrainingSet(m_evalType, vTrainingSet, getAbsNames); 
}

size_t HyperRGBD::Washington::GetTestSet(std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames)
{
	return GetTestSet(m_evalType, vTrainingSet, getAbsNames); 
}

size_t HyperRGBD::Washington::GetAllAbsFilenames(std::vector< std::string > &vAbsFilenames)
{
	const int nMaxCategories = (m_nMaxCategories_test == numeric_limits<int>::max())?m_nMaxCategories_train:max(m_nMaxCategories_train, m_nMaxCategories_test);
	const int nMaxInstances = (m_nMaxInstances_test == numeric_limits<int>::max())?m_nMaxInstances_train:max(m_nMaxInstances_train, m_nMaxInstances_test);
	const int nMaxInstanceViews = (m_nMaxInstanceViews_test == numeric_limits<int>::max())?m_nMaxInstanceViews_train:max(m_nMaxInstanceViews_train, m_nMaxInstanceViews_test);
	return GetAllAbsFilenames(nMaxCategories, nMaxInstances, nMaxInstanceViews, vAbsFilenames );
}


size_t HyperRGBD::Washington::GetTrainingSet_CatRec_Full(vector< pair< string, vector<string> > > &vTrainingSet, const bool getAbsNames)
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
	FindDirsEndWith(m_absEvalDir, "", vCatDirs_rel, true, m_nMaxCategories_train);

	for(size_t ca=0; ca<vCatDirs_rel.size(); ca++)
	{
		cout << "Scan and collect filenames. Cat: " << ca << "/" << vCatDirs_rel.size() << "\r";
		//find all instance dirs in cat dirs
		string catDir_abs = m_absEvalDir + "/" + vCatDirs_rel[ca];

		vector<string> vInstanceDirs_rel;
		FindDirsStartWith(catDir_abs, "", vInstanceDirs_rel, true, m_nMaxInstances_train);


		vector<size_t> vInstBegins(vInstanceDirs_rel.size()+1);

		//collect all files in all instance dirs
		vector<string> vCatSamples_all;
		string suffix = "_crop.png";
		for(size_t in=0; in<vInstanceDirs_rel.size(); in++)
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


size_t HyperRGBD::Washington::GetTrainingSet_InstRec_Full(vector< pair< string, vector<string> > > &vTrainingSet, const bool getAbsNames)
{
	//if(m_trial != 0)
	//{
	//	throw runtime_error("ERROR (RGBDObjectDataset::GetTrainingSet_InstRec_Full): (m_trial != 1)   " + ToString(m_trial));
	//}

	size_t nSamples_tot = 0;
	vTrainingSet.clear();

	//find all category dirs
	
	vector<string> vCatDirs_rel;
	FindDirsEndWith(m_absEvalDir, "", vCatDirs_rel, true, m_nMaxCategories_train);

	for(size_t ca=0; ca<vCatDirs_rel.size(); ca++)
	{
		cout << "Scan and collect filenames. Cat: " << ca << "/" << vCatDirs_rel.size() << "\r";
		//find all instance dirs in cat dirs
		string catDir_abs = m_absEvalDir + "/" + vCatDirs_rel[ca];

		vector<string> vInstanceDirs_rel;
		FindDirsStartWith(catDir_abs, "", vInstanceDirs_rel, true, m_nMaxInstances_train);

		//collect all files in all instance dirs
		
		string suffix = "_crop.png";
		for(size_t in=0; in<vInstanceDirs_rel.size(); in++)
		{
			string instanceDir_abs = catDir_abs + "/" + vInstanceDirs_rel[in];

			//get all files
			vector<string> vCatSamples_Inst;
			FindFilesEndWith(instanceDir_abs, suffix, vCatSamples_Inst, !getAbsNames, m_nMaxInstanceViews_train);
			if(!getAbsNames)
			{
				Prefix_Path(vCatSamples_Inst, vInstanceDirs_rel[in]);
				Prefix_Path(vCatSamples_Inst, vCatDirs_rel[ca]);
			}

			//keep only beginning of filenames
			size_t suffixSize = suffix.size();
			for(size_t sa=0; sa<vCatSamples_Inst.size(); sa++)
			{
				vCatSamples_Inst[sa].resize(vCatSamples_Inst[sa].size() - suffixSize);
			}

			vTrainingSet.push_back( pair< string, vector<string> > (vInstanceDirs_rel[in], vCatSamples_Inst) );
			nSamples_tot += vCatSamples_Inst.size();

		}
	}

	cout << endl;

	return nSamples_tot;

}

