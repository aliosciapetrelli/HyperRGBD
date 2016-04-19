#include "hyperrgbdMVRED.h"
#include "hyperrgbdUtils.h"


using namespace cv;
using namespace std;

void HyperRGBD::MVRED::Init()
{
	m_nMaxCategories_train = numeric_limits<int>::max();
	m_nMaxInstances_train = numeric_limits<int>::max();
	m_nMaxInstanceViews_train = numeric_limits<int>::max();

	m_nMaxCategories_test = numeric_limits<int>::max();
	m_nMaxInstances_test = numeric_limits<int>::max();
	m_nMaxInstanceViews_test = numeric_limits<int>::max();

	m_trial = -1;
	m_is721 = false;

	m_testSetRatio = 1/float(GetNumTrials());

	m_evalType = "";

	m_vCategories.clear();
	PushBackCatVect(m_vCategories,"plane_toy", 1, 3);
	PushBackCatVect(m_vCategories,"apple", 4, 12);
	PushBackCatVect(m_vCategories,"wallet", 13, 20);
	PushBackCatVect(m_vCategories,"banana", 21, 26);
	PushBackCatVect(m_vCategories,"pepper", 27, 32);	//pepper ???
	PushBackCatVect(m_vCategories,"shuttlecock", 33, 33);/*one inst*/
	PushBackCatVect(m_vCategories,"book", 34, 40);
	PushBackCatVect(m_vCategories,"box", 41, 46);
	PushBackCatVect(m_vCategories,"calculator", 47, 50);
	PushBackCatVect(m_vCategories,"hat", 51, 59);			/*different from cap of Bo dataset*/
	PushBackCatVect(m_vCategories,"comb", 60, 64);
	PushBackCatVect(m_vCategories,"cup", 65, 68);
	PushBackCatVect(m_vCategories,"thermos", 69, 70);
	PushBackCatVect(m_vCategories,"matrioska", 71, 71);
	PushBackCatVect(m_vCategories,"dry_battery", 72, 79);
	PushBackCatVect(m_vCategories,"toothpaste", 80, 82);
	//PushBackCatVect(m_vCategories,"???", 83, 83); /*diversa dalle precedenti 3 istanze, cilindrica, incorporare?*/	//sembra un deodorante stick
	PushBackCatVect(m_vCategories,"kleenex", 84, 88);
	PushBackCatVect(m_vCategories,"garlic", 89, 98);
	PushBackCatVect(m_vCategories,"tupperware", 99, 99); 
	PushBackCatVect(m_vCategories,"motorbike_toy", 100, 101);/*toy*/
	PushBackCatVect(m_vCategories,"mushroom", 102, 111);
	PushBackCatVect(m_vCategories,"instant_noodles", 112, 117);
	PushBackCatVect(m_vCategories,"onion", 118, 125);
	PushBackCatVect(m_vCategories,"orange", 126, 135);
	PushBackCatVect(m_vCategories,"peach", 136, 145);
	PushBackCatVect(m_vCategories,"apple", 146, 150);
	PushBackCatVect(m_vCategories,"potato", 151, 153);
	PushBackCatVect(m_vCategories,"trash_bag", 154, 156); /*2 inst*/
	PushBackCatVect(m_vCategories,"scarf", 157, 165);
	PushBackCatVect(m_vCategories,"liquid_soap", 166, 166); /*1 inst*/
	PushBackCatVect(m_vCategories,"shampoo", 167, 167); /*1 inst*/
	PushBackCatVect(m_vCategories,"shark_toy", 168, 172);/*toy*/
	PushBackCatVect(m_vCategories,"shoe", 173, 180);
	PushBackCatVect(m_vCategories,"tomato", 181, 190);
	PushBackCatVect(m_vCategories,"toothbrush", 191, 193);
	PushBackCatVect(m_vCategories,"toothpaste", 194, 200);
	PushBackCatVect(m_vCategories,"peluche", 201, 203);/*toy*/
	PushBackCatVect(m_vCategories,"vehicle_toy", 204, 204);/*toy*/
	PushBackCatVect(m_vCategories,"plane_toy", 205, 205);/*toy*/
	PushBackCatVect(m_vCategories,"apple", 206, 210); /*double*/
	PushBackCatVect(m_vCategories,"wallet", 211, 214); /*double*/
	PushBackCatVect(m_vCategories,"banana", 215, 219); /*double*/
	PushBackCatVect(m_vCategories,"pepper", 220, 228); /*double*/
	PushBackCatVect(m_vCategories,"shuttlecock", 229, 230); /*one inst*/ /*double*/
	PushBackCatVect(m_vCategories,"hair_dryer", 231, 234);
	PushBackCatVect(m_vCategories,"book", 235, 239); /*double*/
	PushBackCatVect(m_vCategories,"pills", 240, 249);
	PushBackCatVect(m_vCategories,"bow_toy", 250, 250); /*toy*/
	PushBackCatVect(m_vCategories,"box", 251, 254);
	PushBackCatVect(m_vCategories,"calculator", 255, 261);
	PushBackCatVect(m_vCategories,"pills", 262, 263); /*double*/
	PushBackCatVect(m_vCategories,"soda_can", 264, 270);
	PushBackCatVect(m_vCategories,"hat", 271, 277); /*sconsiglio merge con i cap di Bo*/
	PushBackCatVect(m_vCategories,"car_toy", 278, 278); /*toy*/
	PushBackCatVect(m_vCategories,"carrot", 279, 287);
	PushBackCatVect(m_vCategories,"cow_toy", 288, 289); /*toy*/
	PushBackCatVect(m_vCategories,"comb", 290, 297);
	PushBackCatVect(m_vCategories,"cup", 298, 299);
	PushBackCatVect(m_vCategories,"eggplant", 300, 304); 
	PushBackCatVect(m_vCategories,"lotion_tube", 305, 307); /*double*/
	PushBackCatVect(m_vCategories,"lotion_tube", 309, 310); /*double*/
	PushBackCatVect(m_vCategories,"liquid_soap", 311, 311);
	PushBackCatVect(m_vCategories,"kleenex", 312, 318); /*double*/
	PushBackCatVect(m_vCategories,"glasses_case", 319, 326);
	PushBackCatVect(m_vCategories,"gun_toy", 327, 328); /*toy*/
	PushBackCatVect(m_vCategories,"helicopter_toy", 329, 330); /*toy*/
	PushBackCatVect(m_vCategories,"horse_toy", 331, 332); /*toy*/
	PushBackCatVect(m_vCategories,"insect_toy", 333, 337); /*toy*/
	PushBackCatVect(m_vCategories,"tupperware", 338, 339); /*double*/
	PushBackCatVect(m_vCategories,"bowl", 340, 342); /*double*/
	PushBackCatVect(m_vCategories,"tupperware", 343, 344); /*double*/
	PushBackCatVect(m_vCategories,"motorbike_toy", 345, 346); /*toy*/ /*double*/
	PushBackCatVect(m_vCategories,"mouse", 347, 353);
	PushBackCatVect(m_vCategories,"mushroom", 354, 358); /*double*/
	PushBackCatVect(m_vCategories,"instant_noodles", 359, 361); /*double*/
	PushBackCatVect(m_vCategories,"onion", 362, 367); /*double*/
	PushBackCatVect(m_vCategories,"orange", 368, 372); /*double*/
	PushBackCatVect(m_vCategories,"peach", 373, 377); /*double*/
	PushBackCatVect(m_vCategories,"apple", 378, 392); /*double*/
	PushBackCatVect(m_vCategories,"pencilcase", 393, 396); 
	PushBackCatVect(m_vCategories,"plant", 397, 404);
	PushBackCatVect(m_vCategories,"power_bank", 405, 415);
	PushBackCatVect(m_vCategories,"potato", 416, 419); /*double*/
	PushBackCatVect(m_vCategories,"pingpong_racket", 420, 423);
	PushBackCatVect(m_vCategories,"scarf", 424, 434); /*double*/
	PushBackCatVect(m_vCategories,"shampoo", 435, 444);
	PushBackCatVect(m_vCategories,"boat_toy", 445, 445); /*toy*/
	PushBackCatVect(m_vCategories,"shoe", 446, 450);				//divide in shoe slipper and sandal
	PushBackCatVect(m_vCategories,"snack", 451, 458);
	PushBackCatVect(m_vCategories,"lamp", 459, 466);
	PushBackCatVect(m_vCategories,"tomato", 467, 471); /*double*/
	PushBackCatVect(m_vCategories,"toothbrush", 472, 473); /*double*/
	PushBackCatVect(m_vCategories,"toothpaste", 474, 481); /*double*/
	PushBackCatVect(m_vCategories,"peluche", 482, 491); /*double*/
	PushBackCatVect(m_vCategories,"vehicle_toy", 492, 493); /*double*/ /*toy*/
	PushBackCatVect(m_vCategories,"umbrella", 494, 507); /*double*/


	for(size_t ca=0; ca<m_vCategories.size(); ca++)
	{
		if(m_vCategories[ca].second.first>m_vCategories[ca].second.second)
		{
			throw runtime_error("ERROR (MVRED::Init): Inverted Interval in category: " + ToString(ca) + ", " + m_vCategories[ca].first + "; instances: " + ToString(m_vCategories[ca].second.first) + " - " + ToString(m_vCategories[ca].second.second) );
		}

		if(ca<m_vCategories.size()-1)
		{
			if(m_vCategories[ca].second.first>m_vCategories[ca+1].second.first)
			{
				throw runtime_error("ERROR (MVRED::Init): Intervals overlapping for the categories: " + ToString(ca) + ", " + ToString(ca+1) + "; " + m_vCategories[ca].first + ", " + m_vCategories[ca+1].first + "; instances: " + ToString(m_vCategories[ca].second.first) + " - " + ToString(m_vCategories[ca+1].second.second) );
			}
		}
	}



}

HyperRGBD::MVRED::MVRED(const string &absDatasetRoot)
{
	Init();

	SetRoot(absDatasetRoot);
}

HyperRGBD::MVRED::MVRED()
{
	Init();
}



size_t HyperRGBD::MVRED::GetTrainingSet_InstRec_Full(std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames, const bool is721)
{
	size_t nSamples_tot = 0;

	size_t picsPerType = 36;
	if (is721 == true){picsPerType = 360;}
	string picsPerInst = ToString(picsPerType*2+1);


	vTrainingSet.clear();
	size_t degChange = 205;

	size_t acqDegrees [3] = {0, 45, 90};
	//string imgType [3] = {"/Depth_Image", "/Mask_white", "/RGB_Image"};
	

	for(size_t in=1; in<=507; in++)
	{
		vector <std::string> vFilenames;

		if ((in == 175) || (in == 388))
		{
			continue;
		}

		if (in == 83)
		{
			continue;
		}

		if (in == 308)
		{
			continue;
		}

		if (in==degChange)
		{
			acqDegrees[0] = 45;
			acqDegrees[1] = 60;
		}

		string instanceName = "t" + ToString(in);

		cout << "Scan and collect. Instance: " << in-1 << "/" << 507 << "\r";

		for(size_t de=0; de<3; de++)
		{
			vector <std::string> vSamples;
			string instRel = picsPerInst + "/" + instanceName + "/Depth_Image" + "/view_" + ToString(acqDegrees[de]);
			string instPath = m_absDatasetRoot + "/" + instRel; 
			FindFilesEndWith(instPath, "", vSamples, !getAbsNames, m_nMaxInstanceViews_train);

			if(!getAbsNames)
			{
				Prefix_Path(vSamples, instRel);
			}

			//remove extension
			for(size_t sa=0; sa<vSamples.size(); sa++)
			{
				vSamples[sa].resize(vSamples[sa].size() - 4);
			}

			vFilenames.insert(vFilenames.end(), vSamples.begin(), vSamples.end() );
			
			nSamples_tot += vSamples.size();

			if( (m_nMaxInstanceViews_train == std::numeric_limits<int>::max()) && ((vSamples.size() != picsPerType) && (vSamples.size() != 1)))
			{
				throw runtime_error("ERROR (MVRED::GetAllAbsFilenames): (vSamples.size() !=" + ToString(picsPerType) + ")" + instPath);
			}
		}

		//string instPath = m_absDatasetRoot + "/" + picsPerInst + instDirs_rel; 
		vTrainingSet.push_back( pair< string, vector<string> > (instanceName, vFilenames) );
	}

	return nSamples_tot;
}

size_t HyperRGBD::MVRED::GetTrainingSet_CatRec_Full(std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames, const bool is721)
{
	size_t picsPerType = 36;
	
	map <std::string, std::vector<std::string> > temp_mTrainingSet;

	if (is721 == true){picsPerType = 360;}
	string picsPerInst = ToString(picsPerType*2+1);


	vTrainingSet.clear();
	size_t degChange = 205;

	size_t acqDegrees [3] = {0, 45, 90};
	//string imgType [3] = {"/Depth_Image", "/Mask_white", "/RGB_Image"};

	size_t nSamples_tot = 0;
	size_t ca = 0;

	for(size_t in=1; in<=507; in++)
	{
		if ((in == 175) || (in == 388))	//not existing
		{
			continue;
		}

		if (in == 83)
		{
			continue;
		}

		if (in == 308)
		{
			continue;
		}

		if (in==degChange)
		{
			acqDegrees[0] = 45;
			acqDegrees[1] = 60;
		}

		if (in>=m_vCategories[ca].second.first && in<=m_vCategories[ca].second.second)
		{
			//label.push_back(m_vCategories[ca].first);
			string instanceName = "t" + ToString(in);

			cout << "Scan and collect. Instance: " << in-1 << "/" << 507 << "\r";

			vector <std::string> vFilenames;

			for(size_t de=0; de<3; de++)
			{
				vector <std::string> vSamples;

				string instRel = picsPerInst + "/" + instanceName + "/Depth_Image" + "/view_" + ToString(acqDegrees[de]);
				string instPath = m_absDatasetRoot + "/" + instRel; 
				FindFilesEndWith(instPath, "", vSamples, !getAbsNames, m_nMaxInstanceViews_train);

				if(!getAbsNames)
				{
					Prefix_Path(vSamples, instRel);
				}

				//remove extension
				for(size_t sa=0; sa<vSamples.size(); sa++)
				{
					vSamples[sa].resize(vSamples[sa].size() - 4);
				}

				nSamples_tot += vSamples.size();

				vFilenames.insert(vFilenames.end(), vSamples.begin(), vSamples.end() );

				if( (m_nMaxInstanceViews_train == std::numeric_limits<int>::max()) && ((vSamples.size() != picsPerType) && (vSamples.size() != 1)))
				{
					throw runtime_error("ERROR (MVRED::GetAllAbsFilenames): (vSamples.size() !=" + ToString(picsPerType) + ")" + instPath);
				}
			}

			vTrainingSet.push_back( pair< string, vector<string> > (m_vCategories[ca].first, vFilenames) );
		}

		if (in==m_vCategories[ca].second.second)
		{
			ca++;
		}

	}


	map<string, vector <size_t> > mapInstBegins;

	for(size_t ap=0; ap < vTrainingSet.size(); ap++)
	{
		if( mapInstBegins.find( vTrainingSet[ap].first ) == mapInstBegins.end() )
		{
			mapInstBegins[ vTrainingSet[ap].first ] = vector <size_t>(1);
			mapInstBegins[ vTrainingSet[ap].first ][0] = 0;
		}
		else
		{
			mapInstBegins[ vTrainingSet[ap].first ].push_back( temp_mTrainingSet[vTrainingSet[ap].first].size() );
		}

		temp_mTrainingSet[vTrainingSet[ap].first].insert(temp_mTrainingSet[vTrainingSet[ap].first].end(), vTrainingSet[ap].second.begin(), vTrainingSet[ap].second.end() );
	}
	//DS_vectToMap(vTrainingSet, temp_mTrainingSet);
	DS_mapToVect(temp_mTrainingSet, vTrainingSet);

	m_vDataSet_InstBegins.clear();
	for(size_t ca=0; ca < vTrainingSet.size(); ca++)
	{
		mapInstBegins[ vTrainingSet[ca].first ].push_back( vTrainingSet[ca].second.size() ); 
	}

	m_vDataSet_InstBegins.resize( vTrainingSet.size() );
	for(size_t ca=0; ca < vTrainingSet.size(); ca++)
	{
		m_vDataSet_InstBegins[ca].first = vTrainingSet[ca].first;
		m_vDataSet_InstBegins[ca].second = mapInstBegins[ vTrainingSet[ca].first ];
	}

	return nSamples_tot;
}

size_t HyperRGBD::MVRED::GetTrainingSet(std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames)
{
	return GetTrainingSet(m_evalType, m_trial, vTrainingSet, getAbsNames, m_is721); 
}

size_t HyperRGBD::MVRED::GetTrainingSet(const string &evalType, const int trial, std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames, const bool is721)
{
	if(m_absDatasetRoot == "")
	{
		cout << "ERROR (GetTrainingSet): m_absDatasetRoot not set";
	}

	if(evalType == "InstRec" || evalType == "InstRec_1EveryN")
	{
		return getTrainingSet_InstRec(trial, vTrainingSet, getAbsNames);
	}
	if(evalType == "CatRec" || evalType == "CatRec_1EveryN")
	{
		return getTrainingSet_CatRec(trial, vTrainingSet, getAbsNames);
	}
	else if(evalType == "InstRec_Full")
	{
		return GetTrainingSet_InstRec_Full(vTrainingSet, getAbsNames, is721);
	}
	else if(evalType == "CatRec_Full")
	{
		return GetTrainingSet_CatRec_Full(vTrainingSet, getAbsNames, is721);
	}
	else
	{
		cout << "ERROR: Uncorrect evalType: " << evalType << endl;
		getchar();
		exit(-1);
	}
}


size_t HyperRGBD::MVRED::GetTestSet(std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames)
{
	return GetTestSet(m_evalType, m_trial, vTrainingSet, getAbsNames, m_is721); 
}

size_t HyperRGBD::MVRED::GetTestSet(const string &evalType, const int trial, std::vector< std::pair< std::string, std::vector<std::string> > > &vTestSet, const bool getAbsNames, const bool is721)
{
	if(m_absDatasetRoot == "")
	{
		cout << "ERROR (GetTestSet): m_absDatasetRoot not set";
	}

	if(evalType == "InstRec" || evalType == "InstRec_1EveryN")
	{
		return getTestSet_InstRec(trial, vTestSet, getAbsNames);
	}
	if(evalType == "CatRec" || evalType == "CatRec_1EveryN")
	{
		return getTestSet_CatRec(trial, vTestSet, getAbsNames);
	}
	else
	{
		cout << "ERROR: Uncorrect evalType: " << evalType << endl;
		getchar();
		exit(-1);
	}

}


string HyperRGBD::MVRED::GetImageFilename_AbsFull(const string &absFilename_withoutSuffix, const ImageType imageType)
{

	string absFilename = absFilename_withoutSuffix;
	string keyWr = "Depth_Image";
	size_t keyWrPos = absFilename.rfind(keyWr);

	if (keyWrPos != std::string::npos)
	{
		switch(imageType)
		{
		/*case IMAGETYPE_AS_IS:

			break;*/
		case IMAGETYPE_AS_IS:
			absFilename = absFilename.replace(keyWrPos, keyWr.length(), "RGB_Image") + ".jpg";
			return absFilename;
			break;
		case IMAGETYPE_RGB:
			absFilename = absFilename.replace(keyWrPos, keyWr.length(), "RGB_Image") + ".jpg";
			return absFilename;
			break;
		case IMAGETYPE_GREY:
			absFilename = absFilename.replace(keyWrPos, keyWr.length(), "RGB_Image") + ".jpg";
			return absFilename;
			break;
		case IMAGETYPE_DEPTHMAP:
			absFilename = absFilename_withoutSuffix + ".png";
			return absFilename;
			break;
		case IMAGETYPE_RANGEMAP:
			absFilename = absFilename_withoutSuffix + ".png";
			return absFilename;
			break;
		case IMAGETYPE_MASK:
			absFilename = absFilename.replace(keyWrPos, keyWr.length(), "Mask_white") + ".jpg";
			return absFilename;		
			break;
		//case IMAGETYPE_POSE:
		//
		//	
		//	break;
		default:
			throw runtime_error("ERROR (RGBDObjectDataset::ReadImage): imageType does not exist");
		}
	}
	else
	{
		throw runtime_error("ERROR: keyword not found");
	}
}

void HyperRGBD::MVRED::GetImageFilename_AbsFull(std::vector< std::pair< std::string, std::vector<std::string> > > &vDataSet, const ImageType imageType)
{
	for(size_t cl=0; cl<vDataSet.size(); cl++)
	{
		for(size_t im=0; im<vDataSet[cl].second.size(); im++)
		{
			vDataSet[cl].second[im] = MVRED::GetImageFilename_AbsFull(vDataSet[cl].second[im], imageType);
		}
	}
}




size_t HyperRGBD::MVRED::GetAllAbsFilenames(std::vector< std::string > &vAbsFilenames)
{
	const int nMaxInstances = (m_nMaxInstances_test == numeric_limits<int>::max())?m_nMaxInstances_train:max(m_nMaxInstances_train, m_nMaxInstances_test);
	//const int nMaxInstances = 10;
	const int nMaxInstanceViews = (m_nMaxInstanceViews_test == numeric_limits<int>::max())?m_nMaxInstanceViews_train:max(m_nMaxInstanceViews_train, m_nMaxInstanceViews_test);
	//const int nMaxInstanceViews = 1;
	return GetAllAbsFilenames(nMaxInstances, nMaxInstanceViews, vAbsFilenames, m_is721);
}

size_t HyperRGBD::MVRED::GetAllAbsFilenames(const int nMaxInstances, const int nMaxInstanceViews, std::vector< std::string > &vAbsFilenames, const bool is721)
{
	vAbsFilenames.clear();
	int acqDegrees [3] = {0, 45, 90};
	int picsPerType = 36;
	if (is721 == true)
	{
		picsPerType = 360;
	}
	string picsPerInst = ToString(picsPerType*2+1);
	
	


	for(size_t in=1; in<508; in++)
	{
		if ((in == 175) || (in == 388))
		{
			continue;
		}

		if (in == 83)
		{
			continue;
		}

		if (in == 308)
		{
			continue;
		}

		if (in >= nMaxInstances)
		{
			break;
		}

		if ((in>204) && (acqDegrees[0] == 0))
		{
			acqDegrees[0] = 45;
			acqDegrees[1] = 60;
		}

		string instDirs_rel = "t" + ToString(in);

		cout << "Scan and collect filenames. Instance: " << in-1 << "/507" "\r";

		vector<string> vSamples;

		for(size_t de=0; de<3; de++)
		{

			string instDir_abs = m_absDatasetRoot + "/" + picsPerInst + "/" + instDirs_rel + "/Depth_Image/view_" + ToString(acqDegrees[de]); 
			FindFilesEndWith(instDir_abs, "png", vSamples, false, nMaxInstanceViews);
			vAbsFilenames.insert(vAbsFilenames.end(), vSamples.begin(), vSamples.end() );


			if( (nMaxInstanceViews == std::numeric_limits<int>::max()) && ((vSamples.size() != picsPerType) && (vSamples.size() != 1)))
			{
				throw runtime_error("ERROR (MVRED::GetAllAbsFilenames): (vSamples.size() !=" + ToString(picsPerType) + ")" + instDir_abs);
			}

		}


		
	}

	//keep only beginning of filenames
	string suffix = ".png";
	size_t suffixSize = suffix.size();
	for(size_t sa=0; sa<vAbsFilenames.size(); sa++)
	{
		vAbsFilenames[sa].resize(vAbsFilenames[sa].size() - suffixSize);
	}

	cout << endl;
	return vAbsFilenames.size();
}




cv::Mat HyperRGBD::MVRED::ReadDepthMap(const std::string &absFilename)
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

cv::Mat HyperRGBD::MVRED::ReadRangeMap(const std::string &absFilename)
{
	cv::Mat depthMap = imread(absFilename, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
	cv::Mat rangeMap = DepthMap2RangeMap(depthMap);

	//cv::imshow("depthMap", depthMap*30);

	//cv::Mat rangeMap_8U;
	//ConvertTo8U(rangeMap, rangeMap_8U);

	//vector<Mat> vComps;
	//cv::split(rangeMap_8U, vComps);
	//cv::imshow("X", vComps[0]);
	//cv::imshow("Y", vComps[1]);
	//cv::imshow("Z", vComps[2]);
	//cv::waitKey(0);

	return rangeMap;
}

cv::Mat HyperRGBD::MVRED::ReadImage(const std::string &absFilename_withoutSuffix, const ImageType imageType)
{
	switch(imageType)
	{
	case IMAGETYPE_AS_IS:
		return imread(GetImageFilename_AbsFull(absFilename_withoutSuffix, imageType), CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH); 
		break;
	case IMAGETYPE_RGB:
		return imread(GetImageFilename_AbsFull(absFilename_withoutSuffix, imageType), CV_LOAD_IMAGE_COLOR | CV_LOAD_IMAGE_ANYDEPTH); 
		break;
	case IMAGETYPE_GREY:
		return imread(GetImageFilename_AbsFull(absFilename_withoutSuffix, imageType), CV_LOAD_IMAGE_GRAYSCALE | CV_LOAD_IMAGE_ANYDEPTH); 
		break;
	case IMAGETYPE_DEPTHMAP:
		return imread(GetImageFilename_AbsFull(absFilename_withoutSuffix, imageType), CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH); 
		break;
	case IMAGETYPE_RANGEMAP:
		return ReadRangeMap(GetImageFilename_AbsFull(absFilename_withoutSuffix, imageType));
		break;
	case IMAGETYPE_MASK:
		return imread(GetImageFilename_AbsFull(absFilename_withoutSuffix, imageType), CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
		break;
	case IMAGETYPE_POSE:
		throw runtime_error("ERROR (RGBDObjectDataset::ReadImage): IMAGETYPE_POSE does not exist for Bo dataset");
		break;
	default:
		throw runtime_error("ERROR (RGBDObjectDataset::ReadImage): imageType does not exist");
	}
}


size_t HyperRGBD::MVRED::getTrainingSet_CatRec(const int trial, vector< pair< string, vector<string> > > &vTrainingSet, const bool getAbsNames)
{
	return getDataSet_CatRec(trial, vTrainingSet, getAbsNames, true, m_nMaxCategories_train, m_nMaxInstances_train, m_nMaxInstanceViews_train);
}

size_t HyperRGBD::MVRED::getTrainingSet_InstRec(const int trial, vector< pair< string, vector<string> > > &vTrainingSet, const bool getAbsNames)
{
	return getDataSet_InstRec(trial, vTrainingSet, getAbsNames, true, m_nMaxCategories_train, m_nMaxInstances_train, m_nMaxInstanceViews_train);
}

size_t HyperRGBD::MVRED::getTestSet_CatRec(const int trial, vector< pair< string, vector<string> > > &vTestSet, const bool getAbsNames)
{
	return getDataSet_CatRec(trial, vTestSet, getAbsNames, false, m_nMaxCategories_test, m_nMaxInstances_test, m_nMaxInstanceViews_test);
}

size_t HyperRGBD::MVRED::getTestSet_InstRec(const int trial, vector< pair< string, vector<string> > > &vTestSet, const bool getAbsNames)
{
	return getDataSet_InstRec(trial, vTestSet, getAbsNames, false, m_nMaxCategories_test, m_nMaxInstances_test, m_nMaxInstanceViews_test);
}




size_t HyperRGBD::MVRED::getDataSet_CatRec(const int trial, vector< pair< string, vector<string> > > &vDataSet, const bool getAbsNames, const bool isTrainingSet, const int nMaxCategories, const int nMaxInstances, const int nMaxInstanceViews)
{

	vector< pair< string, vector<string> > > vDataSet_Full;

	GetTrainingSet_CatRec_Full(vDataSet_Full, getAbsNames, m_is721);


	//get 9/10 for training set or 1/10 for test set
	srand(trial);

	size_t nSamples_tot = 0;
	vDataSet.clear();
	std::vector< std::pair< std::string, std::vector< size_t > > > vDataSet_InstBegins;

	for(size_t ca=0; ca<vDataSet_Full.size() ; ca++)
	{
		string category = vDataSet_Full[ca].first;

		size_t nInstances = m_vDataSet_InstBegins[ca].second.size()-1;
		vector<bool> vInstanceIds(nInstances, false);
		size_t nSampledInstances = Sample_random(vInstanceIds, m_testSetRatio);
		if(nInstances == 1)
		{
			vInstanceIds[0] = !isTrainingSet;
		}
		else
		{
			if(nSampledInstances == 0)
			{
				vInstanceIds[ rand() % vInstanceIds.size()  ] = true;
				nSampledInstances = 1;
			}
		}

		vector<string> vSamples;
		vector<size_t> vInstBegins;

		for(size_t in=0; in<nInstances; in++)
		{	
			if( (isTrainingSet && !vInstanceIds[in]) || (!isTrainingSet && vInstanceIds[in]) )
			{
				vInstBegins.push_back(vSamples.size());
				vSamples.insert(vSamples.end(), vDataSet_Full[ca].second.begin() + m_vDataSet_InstBegins[ca].second[in],  vDataSet_Full[ca].second.begin() + m_vDataSet_InstBegins[ca].second[in+1] );
				
			}
		}
		vInstBegins.push_back(vSamples.size());

		vDataSet.push_back(pair< string, vector<string> > (category, vSamples));
		vDataSet_InstBegins.push_back( pair< string, vector<size_t> > (category, vInstBegins) ); 

		nSamples_tot += vSamples.size();
	}

	m_vDataSet_InstBegins = vDataSet_InstBegins;

	return nSamples_tot;

}

size_t HyperRGBD::MVRED::getDataSet_InstRec(const int trial, vector< pair< string, vector<string> > > &vDataSet, const bool getAbsNames, const bool isTrainingSet, const int nMaxCategories, const int nMaxInstances, const int nMaxInstanceViews)
{
	vector< pair< string, vector<string> > > vDataSet_Full;

	GetTrainingSet_InstRec_Full(vDataSet_Full, getAbsNames, m_is721);


	srand(trial);

	size_t nSamples_tot = 0;
	vDataSet.clear();

	//get 9/10 for training set or 1/10 for test set
	for(size_t in=0; in<vDataSet_Full.size(); in++)
	{

		vector<bool> vTestSampleIds(vDataSet_Full[in].second.size(), false);
		size_t nTestSamples = Sample_random(vTestSampleIds, m_testSetRatio);


		vector<string> vSamples;

		for(size_t sa=0; sa<vTestSampleIds.size(); sa++)
		{
			if( (isTrainingSet && !vTestSampleIds[sa]) || (!isTrainingSet && vTestSampleIds[sa]) )
			{
				vSamples.push_back(vDataSet_Full[in].second[sa]);
			}
		}

		vDataSet.push_back(pair< string, vector<string> > (vDataSet_Full[in].first, vSamples));
		nSamples_tot += vSamples.size();
	}

	return nSamples_tot;
}



void HyperRGBD::MVRED::CropDataset(const std::string &absCroppedDataset, const bool is721, const float bBoxExpansionFactor)
{
	vector< pair< string, vector<string> > > vDataSet_Full;

	GetTrainingSet_InstRec_Full(vDataSet_Full, false, is721);

	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
	compression_params.push_back(100);

	for(size_t in=0; in<vDataSet_Full.size(); in++)
	{
		for(size_t vi=0; vi<vDataSet_Full[in].second.size(); vi++)
		{
			cout << "in: " << in << " - vi: " << vi << "\r";

			Mat mask = ReadImage( m_absDatasetRoot + "/" + vDataSet_Full[in].second[vi], IMAGETYPE_MASK);

			//find Bounding box
			Rect bbMask;
			if(!FindMaskBoundingBox(mask, bbMask, 2.0f)) 
			{
				cout << "  -  No contour" << endl;
				bbMask.x = 200;
				bbMask.y = 150;
				bbMask.width = 265;
				bbMask.height = 188;
			}

			bbMask = Expand(bbMask, bBoxExpansionFactor, mask.cols, mask.rows);

			//crop mask
			Mat mask_cropped;
			mask(bbMask).copyTo(mask_cropped);

			//crop RGB
			Mat rgb = ReadImage( m_absDatasetRoot + "/" + vDataSet_Full[in].second[vi], IMAGETYPE_AS_IS);
			Mat rgb_cropped;
			rgb(bbMask).copyTo(rgb_cropped);

			//crop depth
			bbMask.x -= min(22, bbMask.x);	//manual adjustment due to the not perfect rectification of depth map
			//bbMask.y -= 4;
			Mat depth = ReadImage( m_absDatasetRoot + "/" + vDataSet_Full[in].second[vi], IMAGETYPE_DEPTHMAP);
			Mat depth_cropped;
			depth(bbMask).copyTo(depth_cropped);

			//cv::imshow("Image_depth", depth*30);
			//cv::imshow("Image_rgb", rgb);
			//cv::imshow("Image_mask", mask);

			//cv::imshow("croppedImage_depth", depth_cropped*30);
			//cv::imshow("croppedImage_rgb", rgb_cropped);
			//cv::imshow("croppedImage_mask", mask_cropped);
			//cv::waitKey(1);


			//save images
			string absOut = absCroppedDataset + "/" + GetImageFilename_AbsFull(vDataSet_Full[in].second[vi], IMAGETYPE_AS_IS);
			CreateFullPath( GetPathDir(absOut) );
			imwrite(absOut, rgb_cropped, compression_params);

			absOut = absCroppedDataset + "/" + GetImageFilename_AbsFull(vDataSet_Full[in].second[vi], IMAGETYPE_DEPTHMAP);
			CreateFullPath( GetPathDir(absOut) );
			imwrite(absOut, depth_cropped);

			absOut = absCroppedDataset + "/" + GetImageFilename_AbsFull(vDataSet_Full[in].second[vi], IMAGETYPE_MASK);
			CreateFullPath( GetPathDir(absOut) );
			imwrite(absOut, mask_cropped, compression_params);

		}
	}



}



void HyperRGBD::MVRED::CheckCategories(const bool is721)
{
	vector< pair< string, vector<string> > > vDataSet_Full;

	GetTrainingSet_CatRec_Full(vDataSet_Full, false, m_is721);


	for(size_t ca=0; ca<vDataSet_Full.size() ; ca++)
	{
		string category = vDataSet_Full[ca].first;

		size_t nInstances = m_vDataSet_InstBegins[ca].second.size()-1;

		for(size_t in=0; in<nInstances; in++)
		{	
			
			Mat rgb = ReadImage( m_absDatasetRoot + "/" + vDataSet_Full[ca].second[ m_vDataSet_InstBegins[ca].second[in]   ], IMAGETYPE_AS_IS);
		
			Mat label = GetTextImage(ToString(ca) + " - " + ToString(in) + "/" + ToString(nInstances) + " - " + category, 1, 5);
			Mat filename = GetTextImage(vDataSet_Full[ca].second[ m_vDataSet_InstBegins[ca].second[in]   ], 1, 5);

			cv::imshow("Image_rgb", rgb);
			cv::imshow("label", label);
			cv::imshow("filename", filename);
			cv::waitKey(0);

		}

	}


}