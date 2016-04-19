#include "hyperrgbdHyperDataset.h"
#include "hyperrgbdUtils.h"



using namespace cv;
using namespace std;

void HyperRGBD::HyperDataset::Init()
{
	m_nMaxInstances_train = numeric_limits<int>::max();
	m_nMaxInstanceViews_train = numeric_limits<int>::max();

	m_nMaxInstances_test = numeric_limits<int>::max();
	m_nMaxInstanceViews_test = numeric_limits<int>::max();

	m_trial = -1;

	m_evalType = "";

	m_testSetRatio = 1/float(GetNumTrials());



	for(size_t dt=0; dt<m_datasets.size(); dt++)
	{
		m_datasets[dt].second->Init();
	}

	m_mapCatAssociations["Washington"]["apple"] = "apple";
	m_mapCatAssociations["Washington"]["ball"] = "ball";
	m_mapCatAssociations["Washington"]["banana"] = "banana";
	m_mapCatAssociations["Washington"]["bell_pepper"] = "pepper";
	m_mapCatAssociations["Washington"]["binder"] = "binder";
	m_mapCatAssociations["Washington"]["bowl"] = "bowl";
	m_mapCatAssociations["Washington"]["calculator"] = "calculator";
	m_mapCatAssociations["Washington"]["camera"] = "camera";
	m_mapCatAssociations["Washington"]["cap"] = "hat";
	m_mapCatAssociations["Washington"]["cell_phone"] = "cell_phone";
	m_mapCatAssociations["Washington"]["cereal_box"] = "cereal_box";
	m_mapCatAssociations["Washington"]["coffee_mug"] = "cup";
	m_mapCatAssociations["Washington"]["comb"] = "comb";
	m_mapCatAssociations["Washington"]["dry_battery"] = "dry_battery";
	m_mapCatAssociations["Washington"]["flashlight"] = "flashlight";
	m_mapCatAssociations["Washington"]["food_bag"] = "food_bag";
	m_mapCatAssociations["Washington"]["food_box"] = "food_box";
	m_mapCatAssociations["Washington"]["food_can"] = "food_can";
	m_mapCatAssociations["Washington"]["food_cup"] = "food_cup";
	m_mapCatAssociations["Washington"]["food_jar"] = "food_jar";
	m_mapCatAssociations["Washington"]["garlic"] = "garlic";
	m_mapCatAssociations["Washington"]["glue_stick"] = "glue_stick";
	m_mapCatAssociations["Washington"]["greens"] = "greens";
	m_mapCatAssociations["Washington"]["hand_towel"] = "hand_towel";
	m_mapCatAssociations["Washington"]["instant_noodles"] = "instant_noodles";
	m_mapCatAssociations["Washington"]["keyboard"] = "keyboard";
	m_mapCatAssociations["Washington"]["kleenex"] = "kleenex";
	m_mapCatAssociations["Washington"]["lemon"] = "lemon";
	m_mapCatAssociations["Washington"]["lightbulb"] = "lightbulb";
	m_mapCatAssociations["Washington"]["lime"] = "lime";
	m_mapCatAssociations["Washington"]["marker"] = "marker";
	m_mapCatAssociations["Washington"]["mushroom"] = "mushroom";
	m_mapCatAssociations["Washington"]["notebook"] = "notebook";

	m_mapCatAssociations["Washington"]["onion"] = "onion";
	m_mapCatAssociations["Washington"]["orange"] = "orange";
	m_mapCatAssociations["Washington"]["peach"] = "peach";
	m_mapCatAssociations["Washington"]["pear"] = "pear";
	m_mapCatAssociations["Washington"]["pitcher"] = "pitcher";
	m_mapCatAssociations["Washington"]["plate"] = "plate";
	m_mapCatAssociations["Washington"]["pliers"] = "pliers";
	m_mapCatAssociations["Washington"]["potato"] = "potato";
	m_mapCatAssociations["Washington"]["rubber_eraser"] = "rubber_eraser";
	m_mapCatAssociations["Washington"]["scissors"] = "scissors";
	m_mapCatAssociations["Washington"]["shampoo"] = "shampoo";
	m_mapCatAssociations["Washington"]["soda_can"] = "soda_can";
	m_mapCatAssociations["Washington"]["sponge"] = "sponge";
	m_mapCatAssociations["Washington"]["stapler"] = "stapler";
	m_mapCatAssociations["Washington"]["tomato"] = "tomato";
	m_mapCatAssociations["Washington"]["toothbrush"] = "toothbrush";
	m_mapCatAssociations["Washington"]["toothpaste"] = "toothpaste";
	m_mapCatAssociations["Washington"]["water_bottle"] = "water_bottle";

	m_mapCatAssociations["CIN2D3D"]["Binder"] = "binder";
	m_mapCatAssociations["CIN2D3D"]["Book"] = "book";
	m_mapCatAssociations["CIN2D3D"]["Bottle"] = "bottle";
	m_mapCatAssociations["CIN2D3D"]["Cans"] = "food_can";
	m_mapCatAssociations["CIN2D3D"]["CoffeePot"] = "coffee_pot";
	m_mapCatAssociations["CIN2D3D"]["Cup"] = "cup";
	m_mapCatAssociations["CIN2D3D"]["DishLiquid"] = "dish_liquid";
	m_mapCatAssociations["CIN2D3D"]["DrinkCarton"] = "drink_carton";
	m_mapCatAssociations["CIN2D3D"]["Fork"] = "fork";
	m_mapCatAssociations["CIN2D3D"]["Knife"] = "knife";
	m_mapCatAssociations["CIN2D3D"]["Monitor"] = "monitor";
	m_mapCatAssociations["CIN2D3D"]["Mouse"] = "mouse";
	m_mapCatAssociations["CIN2D3D"]["Pen"] = "pen";
	m_mapCatAssociations["CIN2D3D"]["Perforator"] = "perforator";
	m_mapCatAssociations["CIN2D3D"]["Phone"] = "phone";
	m_mapCatAssociations["CIN2D3D"]["Plate"] = "plate";
	m_mapCatAssociations["CIN2D3D"]["Scissors"] = "scissors";
	m_mapCatAssociations["CIN2D3D"]["Spoon"] = "spoon";

	m_mapCatAssociations["MVRED"]["apple"] = "apple";
	m_mapCatAssociations["MVRED"]["banana"] = "banana";
	m_mapCatAssociations["MVRED"]["boat_toy"] = "boat_toy";
	m_mapCatAssociations["MVRED"]["dry_battery"] = "dry_battery";
	m_mapCatAssociations["MVRED"]["book"] = "book";
	m_mapCatAssociations["MVRED"]["bowl"] = "bowl";
	m_mapCatAssociations["MVRED"]["bow_toy"] = "bow_toy";
	m_mapCatAssociations["MVRED"]["box"] = "box";
	m_mapCatAssociations["MVRED"]["calculator"] = "calculator";
	m_mapCatAssociations["MVRED"]["car_toy"] = "car_toy";
	m_mapCatAssociations["MVRED"]["carrot"] = "carrot";
	m_mapCatAssociations["MVRED"]["comb"] = "comb";
	m_mapCatAssociations["MVRED"]["cow_toy"] = "cow_toy";
	m_mapCatAssociations["MVRED"]["cup"] = "cup";
	m_mapCatAssociations["MVRED"]["eggplant"] = "eggplant";
	m_mapCatAssociations["MVRED"]["lamp"] = "lamp";
	m_mapCatAssociations["MVRED"]["garlic"] = "garlic";
	m_mapCatAssociations["MVRED"]["glasses_case"] = "glasses_case";
	m_mapCatAssociations["MVRED"]["gun_toy"] = "gun_toy";
	m_mapCatAssociations["MVRED"]["hair_dryer"] = "hair_dryer";
	m_mapCatAssociations["MVRED"]["hat"] = "hat";
	m_mapCatAssociations["MVRED"]["helicopter_toy"] = "helicopter_toy";
	m_mapCatAssociations["MVRED"]["horse_toy"] = "horse_toy";
	m_mapCatAssociations["MVRED"]["insect_toy"] = "insect_toy";
	m_mapCatAssociations["MVRED"]["instant_noodles"] = "instant_noodles";
	m_mapCatAssociations["MVRED"]["kleenex"] = "kleenex";
	m_mapCatAssociations["MVRED"]["liquid_soap"] = "liquid_soap";
	m_mapCatAssociations["MVRED"]["lotion_tube"] = "lotion_tube";
	m_mapCatAssociations["MVRED"]["matrioska"] = "matrioska";
	m_mapCatAssociations["MVRED"]["motorbike_toy"] = "motorbike_toy";
	m_mapCatAssociations["MVRED"]["mouse"] = "mouse";
	m_mapCatAssociations["MVRED"]["mushroom"] = "mushroom";
	m_mapCatAssociations["MVRED"]["onion"] = "onion";
	m_mapCatAssociations["MVRED"]["orange"] = "orange";
	m_mapCatAssociations["MVRED"]["peach"] = "peach";
	m_mapCatAssociations["MVRED"]["peluche"] = "peluche";
	m_mapCatAssociations["MVRED"]["pencilcase"] = "pencilcase";
	m_mapCatAssociations["MVRED"]["pepper"] = "pepper";
	m_mapCatAssociations["MVRED"]["pills"] = "pills";
	m_mapCatAssociations["MVRED"]["pingpong_racket"] = "pingpong_racket";
	m_mapCatAssociations["MVRED"]["plane_toy"] = "plane_toy";
	m_mapCatAssociations["MVRED"]["plant"] = "plant";
	m_mapCatAssociations["MVRED"]["potato"] = "potato";
	m_mapCatAssociations["MVRED"]["power_bank"] = "power_bank";
	m_mapCatAssociations["MVRED"]["scarf"] = "scarf";
	m_mapCatAssociations["MVRED"]["shampoo"] = "shampoo";
	m_mapCatAssociations["MVRED"]["shark_toy"] = "shark_toy";
	m_mapCatAssociations["MVRED"]["shoe"] = "shoe";
	m_mapCatAssociations["MVRED"]["shuttlecock"] = "shuttlecock";
	m_mapCatAssociations["MVRED"]["snack"] = "snack";
	m_mapCatAssociations["MVRED"]["soda_can"] = "soda_can";
	m_mapCatAssociations["MVRED"]["thermos"] = "thermos";
	m_mapCatAssociations["MVRED"]["tomato"] = "tomato";
	m_mapCatAssociations["MVRED"]["toothbrush"] = "toothbrush";
	m_mapCatAssociations["MVRED"]["toothpaste"] = "toothpaste";
	m_mapCatAssociations["MVRED"]["trash_bag"] = "trash_bag";
	m_mapCatAssociations["MVRED"]["tupperware"] = "tupperware";
	m_mapCatAssociations["MVRED"]["umbrella"] = "umbrella";
	m_mapCatAssociations["MVRED"]["vehicle_toy"] = "vehicle_toy";
	m_mapCatAssociations["MVRED"]["wallet"] = "wallet";

	m_mapCatAssociations["CroppedMVRED"]["apple"] = "apple";
	m_mapCatAssociations["CroppedMVRED"]["banana"] = "banana";
	m_mapCatAssociations["CroppedMVRED"]["boat_toy"] = "boat_toy";
	m_mapCatAssociations["CroppedMVRED"]["dry_battery"] = "dry_battery";
	m_mapCatAssociations["CroppedMVRED"]["book"] = "book";
	m_mapCatAssociations["CroppedMVRED"]["bowl"] = "bowl";
	m_mapCatAssociations["CroppedMVRED"]["bow_toy"] = "bow_toy";
	m_mapCatAssociations["CroppedMVRED"]["box"] = "box";
	m_mapCatAssociations["CroppedMVRED"]["calculator"] = "calculator";
	m_mapCatAssociations["CroppedMVRED"]["car_toy"] = "car_toy";
	m_mapCatAssociations["CroppedMVRED"]["carrot"] = "carrot";
	m_mapCatAssociations["CroppedMVRED"]["comb"] = "comb";
	m_mapCatAssociations["CroppedMVRED"]["cow_toy"] = "cow_toy";
	m_mapCatAssociations["CroppedMVRED"]["cup"] = "cup";
	m_mapCatAssociations["CroppedMVRED"]["eggplant"] = "eggplant";
	m_mapCatAssociations["CroppedMVRED"]["lamp"] = "lamp";
	m_mapCatAssociations["CroppedMVRED"]["garlic"] = "garlic";
	m_mapCatAssociations["CroppedMVRED"]["glasses_case"] = "glasses_case";
	m_mapCatAssociations["CroppedMVRED"]["gun_toy"] = "gun_toy";
	m_mapCatAssociations["CroppedMVRED"]["hair_dryer"] = "hair_dryer";
	m_mapCatAssociations["CroppedMVRED"]["hat"] = "hat";
	m_mapCatAssociations["CroppedMVRED"]["helicopter_toy"] = "helicopter_toy";
	m_mapCatAssociations["CroppedMVRED"]["horse_toy"] = "horse_toy";
	m_mapCatAssociations["CroppedMVRED"]["insect_toy"] = "insect_toy";
	m_mapCatAssociations["CroppedMVRED"]["instant_noodles"] = "instant_noodles";
	m_mapCatAssociations["CroppedMVRED"]["kleenex"] = "kleenex";
	m_mapCatAssociations["CroppedMVRED"]["liquid_soap"] = "liquid_soap";
	m_mapCatAssociations["CroppedMVRED"]["lotion_tube"] = "lotion_tube";
	m_mapCatAssociations["CroppedMVRED"]["matrioska"] = "matrioska";
	m_mapCatAssociations["CroppedMVRED"]["motorbike_toy"] = "motorbike_toy";
	m_mapCatAssociations["CroppedMVRED"]["mouse"] = "mouse";
	m_mapCatAssociations["CroppedMVRED"]["mushroom"] = "mushroom";
	m_mapCatAssociations["CroppedMVRED"]["onion"] = "onion";
	m_mapCatAssociations["CroppedMVRED"]["orange"] = "orange";
	m_mapCatAssociations["CroppedMVRED"]["peach"] = "peach";
	m_mapCatAssociations["CroppedMVRED"]["peluche"] = "peluche";
	m_mapCatAssociations["CroppedMVRED"]["pencilcase"] = "pencilcase";
	m_mapCatAssociations["CroppedMVRED"]["pepper"] = "pepper";
	m_mapCatAssociations["CroppedMVRED"]["pills"] = "pills";
	m_mapCatAssociations["CroppedMVRED"]["pingpong_racket"] = "pingpong_racket";
	m_mapCatAssociations["CroppedMVRED"]["plane_toy"] = "plane_toy";
	m_mapCatAssociations["CroppedMVRED"]["plant"] = "plant";
	m_mapCatAssociations["CroppedMVRED"]["potato"] = "potato";
	m_mapCatAssociations["CroppedMVRED"]["power_bank"] = "power_bank";
	m_mapCatAssociations["CroppedMVRED"]["scarf"] = "scarf";
	m_mapCatAssociations["CroppedMVRED"]["shampoo"] = "shampoo";
	m_mapCatAssociations["CroppedMVRED"]["shark_toy"] = "shark_toy";
	m_mapCatAssociations["CroppedMVRED"]["shoe"] = "shoe";
	m_mapCatAssociations["CroppedMVRED"]["shuttlecock"] = "shuttlecock";
	m_mapCatAssociations["CroppedMVRED"]["snack"] = "snack";
	m_mapCatAssociations["CroppedMVRED"]["soda_can"] = "soda_can";
	m_mapCatAssociations["CroppedMVRED"]["thermos"] = "thermos";
	m_mapCatAssociations["CroppedMVRED"]["tomato"] = "tomato";
	m_mapCatAssociations["CroppedMVRED"]["toothbrush"] = "toothbrush";
	m_mapCatAssociations["CroppedMVRED"]["toothpaste"] = "toothpaste";
	m_mapCatAssociations["CroppedMVRED"]["trash_bag"] = "trash_bag";
	m_mapCatAssociations["CroppedMVRED"]["tupperware"] = "tupperware";
	m_mapCatAssociations["CroppedMVRED"]["umbrella"] = "umbrella";
	m_mapCatAssociations["CroppedMVRED"]["vehicle_toy"] = "vehicle_toy";
	m_mapCatAssociations["CroppedMVRED"]["wallet"] = "wallet";

}

HyperRGBD::HyperDataset::HyperDataset()
{
	m_datasets.push_back(pair<string, IDataset*>("Washington", new Washington()) );
	m_datasets.push_back(pair<string, IDataset*>("CIN2D3D", new CIN2D3D()) );
	m_datasets.push_back(pair<string, IDataset*>("CroppedBIRD", new CroppedBIRD()) );
	m_datasets.push_back(pair<string, IDataset*>("CroppedMVRED", new CroppedMVRED()) );

	Init();

}


size_t HyperRGBD::HyperDataset::GetTrainingSet_AllDatasets(const std::string &evalType, const bool getAbsNames, std::vector<std::string> &vNames, std::vector < std::vector < std::pair< std::string, std::vector< std::string> > > > &vvDataSets)
{
	std::vector < std::vector < std::pair< std::string, std::vector< size_t> > > > vvDataSet_InstBegins;
	return GetTrainingSet_AllDatasets(evalType, getAbsNames, vNames, vvDataSets, vvDataSet_InstBegins);
}

size_t HyperRGBD::HyperDataset::GetTrainingSet_AllDatasets(const std::string &evalType, const bool getAbsNames, std::vector<std::string> &vNames, std::vector < std::vector < std::pair< std::string, std::vector< std::string> > > > &vvDataSets, std::vector < std::vector < std::pair< std::string, std::vector< size_t> > > > &vvDataSet_InstBegins)
{
	vNames.clear();
	vvDataSets.clear();
	vvDataSet_InstBegins.clear();

	for(size_t dt=0; dt<m_datasets.size() ; dt++)
	{
		string temp_evalType = m_datasets[dt].second->m_evalType;
		m_datasets[dt].second->m_evalType = evalType;

		vector <pair <string, vector <string> > > vTrainingSet_Temp;

		try
		{
			m_datasets[dt].second->GetTrainingSet(vTrainingSet_Temp, getAbsNames); 
		}
		catch (exception &)
		{
			m_datasets[dt].second->m_evalType = temp_evalType;
			cout << "!! WARNING: " << m_datasets[dt].first << " does not define evalType: " << m_datasets[dt].second->m_evalType << endl;
			continue;
		}

		vNames.push_back( m_datasets[dt].first );

		vvDataSet_InstBegins.push_back( m_datasets[dt].second->GetInstBegins() );

		vvDataSets.push_back(vTrainingSet_Temp);

		m_datasets[dt].second->m_evalType = temp_evalType;
	}

	return vvDataSets.size();
}



size_t HyperRGBD::HyperDataset::GetTrainingSet(std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames)
{
	return GetTrainingSet(m_evalType, m_trial, vTrainingSet, getAbsNames); 
}

size_t HyperRGBD::HyperDataset::GetTrainingSet(const std::string &evalType, const int trial, std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames)
{
	if(evalType == "CatRec_Full")
	{
		return GetTrainingSet_CatRec_Full(vTrainingSet, getAbsNames);
	}
	else if(evalType == "InstRec_Full")
	{
		return GetTrainingSet_InstRec_Full(vTrainingSet, getAbsNames);
	}
	else if(evalType == "InstRec")
	{
		return getTrainingSet_InstRec(trial, vTrainingSet, getAbsNames);
	}
	else if(evalType == "InstRec_Balanced")
	{
		return getTrainingSet_InstRec_Balanced(trial, vTrainingSet, getAbsNames);
	}
	else if(evalType == "CatRec")
	{
		return getTrainingSet_CatRec(trial, vTrainingSet, getAbsNames);
	}
	else if(evalType == "CatRec_Balanced")
	{
		return getTrainingSet_CatRec_Balanced(trial, vTrainingSet, getAbsNames);
	}
	else
	{
		cout << "ERROR: Uncorrect evalType: " << evalType << endl;
		getchar();
		exit(-1);
	}
}

size_t HyperRGBD::HyperDataset::GetTrainingSet_CatRec_Full(std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames)
{
	//get all datasets
	vector<string> vDataSetNames;
	vector < vector < pair< string, vector<string> > > > vvDataSet_All;
	vector < vector < pair< string, vector< size_t > > > > vvDataSet_InstBegins_All;
	GetTrainingSet_AllDatasets("CatRec_Full", getAbsNames, vDataSetNames, vvDataSet_All, vvDataSet_InstBegins_All);


	map<string, vector <string> > mapTrainingSet;
	
	for(size_t dt=0; dt<vDataSetNames.size() ; dt++)
	{
		string dsName = vDataSetNames[dt];

		for(size_t ca=0; ca<vvDataSet_All[dt].size(); ca++)
		{		
			string listedName = vvDataSet_All[dt][ca].first;
			string superCat = m_mapCatAssociations[dsName][listedName];
			mapTrainingSet[superCat].insert(mapTrainingSet[superCat].end(), vvDataSet_All[dt][ca].second.begin(), vvDataSet_All[dt][ca].second.end()) ;
		}

		cout << endl;
	}
	
	DS_mapToVect(mapTrainingSet, vTrainingSet);


	return 0;
}


size_t HyperRGBD::HyperDataset::GetTrainingSet_InstRec_Full(std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames)
{

	//get all datasets
	vector<string> vDataSetNames;
	vector < vector < pair< string, vector<string> > > > vvDataSet_All;
	GetTrainingSet_AllDatasets("InstRec_Full", getAbsNames, vDataSetNames, vvDataSet_All);


	//prefix the name of the dataset to each instance name
	for(size_t dt=0; dt<vvDataSet_All.size() ; dt++)
	{
		for(size_t in=0; in<vvDataSet_All[dt].size() ; in++)
		{
			vvDataSet_All[dt][in].first =vDataSetNames[dt]  + "_" +  vvDataSet_All[dt][in].first; 			
		}
	}


	for(size_t dt=0; dt<vvDataSet_All.size() ; dt++)
	{
		vTrainingSet.insert(vTrainingSet.end(), vvDataSet_All[dt].begin(), vvDataSet_All[dt].end() );
	}

	return vTrainingSet.size();
}


size_t HyperRGBD::HyperDataset::getTrainingSet_CatRec(const int trial, vector< pair< string, vector<string> > > &vTrainingSet, const bool getAbsNames)
{
	return getDataSet_CatRec(trial, vTrainingSet, getAbsNames, true, m_nMaxCategories_train, m_nMaxInstances_train, m_nMaxInstanceViews_train);
}

size_t HyperRGBD::HyperDataset::getTrainingSet_InstRec(const int trial, vector< pair< string, vector<string> > > &vTrainingSet, const bool getAbsNames)
{
	return getDataSet_InstRec(trial, vTrainingSet, getAbsNames, true, m_nMaxCategories_train, m_nMaxInstances_train, m_nMaxInstanceViews_train);
}

size_t HyperRGBD::HyperDataset::GetTestSet(std::vector< std::pair< std::string, std::vector<std::string> > > &vTestSet, const bool getAbsNames)
{
	return GetTestSet(m_evalType, m_trial, vTestSet, getAbsNames); 
}

size_t HyperRGBD::HyperDataset::GetTestSet(const std::string &evalType, const int trial, std::vector< std::pair< std::string, std::vector<std::string> > > &vTestSet, const bool getAbsNames)
{

	if(evalType == "InstRec") 
	{
		return getTestSet_InstRec(trial, vTestSet, getAbsNames);
	}
	else if(evalType == "InstRec_Balanced")
	{
		return getTestSet_InstRec_Balanced(trial, vTestSet, getAbsNames);
	}
	if(evalType == "CatRec") 
	{
		return getTestSet_CatRec(trial, vTestSet, getAbsNames);
	}
	else if(evalType == "CatRec_Balanced")
	{
		return getTestSet_CatRec_Balanced(trial, vTestSet, getAbsNames);
	}
	else
	{
		cout << "ERROR: Uncorrect evalType: " << evalType << endl;
		getchar();
		exit(-1);
	}
}

size_t HyperRGBD::HyperDataset::getTestSet_CatRec(const int trial, vector< pair< string, vector<string> > > &vTestSet, const bool getAbsNames)
{
	return getDataSet_CatRec(trial, vTestSet, getAbsNames, false, m_nMaxCategories_test, m_nMaxInstances_test, m_nMaxInstanceViews_test);
}

size_t HyperRGBD::HyperDataset::getTestSet_InstRec(const int trial, vector< pair< string, vector<string> > > &vTestSet, const bool getAbsNames)
{
	return getDataSet_InstRec(trial, vTestSet, getAbsNames, false, m_nMaxCategories_test, m_nMaxInstances_test, m_nMaxInstanceViews_test);
}


size_t HyperRGBD::HyperDataset::getDataSet_InstRec(const int trial, vector< pair< string, vector<string> > > &vDataSet, const bool getAbsNames, const bool isTrainingSet, const int nMaxCategories, const int nMaxInstances, const int nMaxInstanceViews)
{
	vector< pair< string, vector<string> > > vDataSet_Full;

	GetTrainingSet_InstRec_Full(vDataSet_Full, true);

	size_t nTotSamples = CountDatasetSamples(vDataSet_Full);

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


size_t HyperRGBD::HyperDataset::getDataSet_InstRec_Balanced(const int trial, std::vector< std::pair< std::string, std::vector<std::string> > > &vDataSet, const bool getAbsNames, const bool isTrainingSet, const int nMaxCategories, const int nMaxInstances, const int nMaxInstanceViews)
{
	//get all datasets
	vector<string> vDataSetNames;
	vector < vector < pair< string, vector<string> > > > vvDataSet_All;
	GetTrainingSet_AllDatasets("InstRec_Full", true, vDataSetNames, vvDataSet_All);


	//prefix the name of the dataset to each instance name
	for(size_t dt=0; dt<vvDataSet_All.size() ; dt++)
	{
		for(size_t in=0; in<vvDataSet_All[dt].size() ; in++)
		{
			vvDataSet_All[dt][in].first =vDataSetNames[dt] + "_" +  vvDataSet_All[dt][in].first; 			
		}
	}


	//determine the smallest dataset
	size_t minNumInstances = numeric_limits<size_t>::max();
	for(size_t dt=0; dt<vvDataSet_All.size() ; dt++)
	{
		minNumInstances = min(minNumInstances, vvDataSet_All[dt].size());
	}

	srand(trial);

	size_t nSamples_tot = 0;
	vDataSet.clear();

	
	for(size_t dt=0; dt<vvDataSet_All.size() ; dt++)
	{
		//select minNumInstances instance
		vector<bool> vInstancesIds(vvDataSet_All[dt].size(), false);
		size_t nInstances = Sample_random(vInstancesIds, minNumInstances/(float)vInstancesIds.size());

		for(size_t in=0; in<vvDataSet_All[dt].size(); in++)
		{
			if(!vInstancesIds[in])
			{
				continue;
			}

			//get 9/10 for training set or 1/10 for test set
			vector<bool> vTestSampleIds(vvDataSet_All[dt][in].second.size(), false);
			size_t nTestSamples = Sample_random(vTestSampleIds, m_testSetRatio);

			vector<string> vSamples;

			for(size_t vi=0; vi<vTestSampleIds.size(); vi++)
			{
				if( (isTrainingSet && !vTestSampleIds[vi]) || (!isTrainingSet && vTestSampleIds[vi]) )
				{
					vSamples.push_back(vvDataSet_All[dt][in].second[vi]);
				}
			}

			vDataSet.push_back(pair< string, vector<string> > (vvDataSet_All[dt][in].first, vSamples));
			nSamples_tot += vSamples.size();
		}
	}

	return nSamples_tot;
}

size_t HyperRGBD::HyperDataset::getTrainingSet_InstRec_Balanced(const int trial, vector< pair< string, vector<string> > > &vTrainingSet, const bool getAbsNames)
{
	return getDataSet_InstRec_Balanced(trial, vTrainingSet, getAbsNames, true, m_nMaxCategories_train, m_nMaxInstances_train, m_nMaxInstanceViews_train);
}

size_t HyperRGBD::HyperDataset::getTestSet_InstRec_Balanced(const int trial, vector< pair< string, vector<string> > > &vTestSet, const bool getAbsNames)
{
	return getDataSet_InstRec_Balanced(trial, vTestSet, getAbsNames, false, m_nMaxCategories_test, m_nMaxInstances_test, m_nMaxInstanceViews_test);
}

size_t HyperRGBD::HyperDataset::getTrainingSet_CatRec_Balanced(const int trial, vector< pair< string, vector<string> > > &vTrainingSet, const bool getAbsNames)
{
	return getDataSet_CatRec_Balanced(trial, vTrainingSet, getAbsNames, true, m_nMaxCategories_train, m_nMaxInstances_train, m_nMaxInstanceViews_train);
}

size_t HyperRGBD::HyperDataset::getTestSet_CatRec_Balanced(const int trial, vector< pair< string, vector<string> > > &vTestSet, const bool getAbsNames)
{
	return getDataSet_CatRec_Balanced(trial, vTestSet, getAbsNames, false, m_nMaxCategories_test, m_nMaxInstances_test, m_nMaxInstanceViews_test);
}

size_t HyperRGBD::HyperDataset::getDataSet_CatRec(const int trial, vector< pair< string, vector<string> > > &vDataSet, const bool getAbsNames, const bool isTrainingSet, const int nMaxCategories, const int nMaxInstances, const int nMaxInstanceViews)
{
	//get all datasets
	vector<string> vDataSetNames;
	vector < vector < pair< string, vector<string> > > > vvDataSet_All;
	vector < vector < pair< string, vector< size_t > > > > vvDataSet_InstBegins_All;
	GetTrainingSet_AllDatasets("CatRec_Full", true, vDataSetNames, vvDataSet_All, vvDataSet_InstBegins_All);




	//merging of the datasets (beside mapInstBegins, it is similar to GetTrainingSet_Full)
	map<string, vector <string> > mapDataSet;
	map<string, vector <size_t> > mapInstBegins;

	for(size_t dt=0; dt<vvDataSet_All.size() ; dt++)
	{
		string dsName = vDataSetNames[dt];

		for(size_t ca=0; ca<vvDataSet_All[dt].size(); ca++)
		{
			string listedName = vvDataSet_All[dt][ca].first;
			string superCat = m_mapCatAssociations[dsName][listedName];

			mapDataSet[superCat].insert(mapDataSet[superCat].end(), vvDataSet_All[dt][ca].second.begin(), vvDataSet_All[dt][ca].second.end() ) ;

			size_t lastBegin = 0;
			if(mapInstBegins.find(superCat) != mapInstBegins.end())
			{
				lastBegin = mapInstBegins[superCat][ mapInstBegins[superCat].size()-1 ];
				mapInstBegins[superCat].resize(	mapInstBegins[superCat].size()-1 );
			}
			vector <size_t> vInstanceBegins = vvDataSet_InstBegins_All[dt][ca].second;
			for(size_t in=0; in<vInstanceBegins.size(); in++)
			{
				vInstanceBegins[in] += lastBegin;
			}
			mapInstBegins[superCat].insert( mapInstBegins[superCat].end(), vInstanceBegins.begin(), vInstanceBegins.end() );
		}
	}

	vector< pair< string, vector<string> > > vDataSet_Full;
	DS_mapToVect(mapDataSet, vDataSet_Full);

	size_t nTotSamples = CountDatasetSamples(vDataSet_Full);

	//get 9/10 for training set or 1/10 for test set
	srand(trial);

	size_t nSamples_tot = 0;
	vDataSet.clear();
	m_vDataSet_InstBegins.clear();

	for(size_t ca=0; ca<vDataSet_Full.size() ; ca++)
	{
		string category = vDataSet_Full[ca].first;

		size_t nInstances = mapInstBegins[category].size()-1;
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
				vSamples.insert(vSamples.end(), vDataSet_Full[ca].second.begin() + mapInstBegins[category][in],  vDataSet_Full[ca].second.begin() + mapInstBegins[category][in+1] );
			}
		}

		vInstBegins.push_back(vSamples.size());

		vDataSet.push_back(pair< string, vector<string> > (category, vSamples));
		m_vDataSet_InstBegins.push_back( pair< string, vector<size_t> > (category, vInstBegins) ); 
		nSamples_tot += vSamples.size();
	}

	return nSamples_tot;
}



size_t HyperRGBD::HyperDataset::getDataSet_CatRec_Balanced(const int trial, vector< pair< string, vector<string> > > &vDataSet, const bool getAbsNames, const bool isTrainingSet, const int nMaxCategories, const int nMaxInstances, const int nMaxInstanceViews)
{
	//get all datasets
	vector<string> vDataSetNames;
	vector < vector < pair< string, vector<string> > > > vvDataSet_All;
	vector < vector < pair< string, vector< size_t > > > > vvDataSet_InstBegins_All;
	GetTrainingSet_AllDatasets("CatRec_Full", true, vDataSetNames, vvDataSet_All, vvDataSet_InstBegins_All);

	//find min num of instances of each supercategory
	map<string, size_t > mapMinNumInstances;
	for(size_t dt=0; dt<vvDataSet_All.size() ; dt++)
	{
		string dsName = vDataSetNames[dt];

		for(size_t ca=0; ca<vvDataSet_All[dt].size(); ca++)
		{
			string listedName = vvDataSet_All[dt][ca].first;
			string superCat = m_mapCatAssociations[dsName][listedName];

			if(superCat == "")
			{
				throw runtime_error("ERROR (getDataSet_CatRec_Balanced): (superCat == "") dsName: " + dsName + " - listedName:" + listedName );
			}

			if(mapMinNumInstances.find(superCat) == mapMinNumInstances.end())
			{
				mapMinNumInstances[superCat] = vvDataSet_InstBegins_All[dt][ca].second.size() - 1;
			}
			else
			{
				mapMinNumInstances[superCat] = min( mapMinNumInstances[superCat], vvDataSet_InstBegins_All[dt][ca].second.size() - 1 );
			}
		}
	}

	srand(trial);

	//merging of the datasets (beside mapInstBegins, it is similar to GetTrainingSet_Full)
	map<string, vector <string> > mapDataSet;
	map<string, vector <size_t> > mapInstBegins;

	for(size_t dt=0; dt<vvDataSet_All.size() ; dt++)
	{
		string dsName = vDataSetNames[dt];

		for(size_t ca=0; ca<vvDataSet_All[dt].size(); ca++)
		{
			string listedName = vvDataSet_All[dt][ca].first;
			string superCat = m_mapCatAssociations[dsName][listedName];

			//select min num instances
			size_t nInstances = vvDataSet_InstBegins_All[dt][ca].second.size() - 1;
			vector<bool> vInstanceIds(nInstances, false);
			size_t nSampledInstances = Sample_random(vInstanceIds, mapMinNumInstances[superCat]/(float)nInstances );
			if(nSampledInstances == 0)
			{
				vInstanceIds[ rand() % vInstanceIds.size()  ] = true;
				nSampledInstances = 1;
			}

			vector<string> vSamples;
			vector<size_t> vInstanceBegins;
			for(size_t in=0; in<nInstances; in++)
			{	
				if( vInstanceIds[in])
				{
					vInstanceBegins.push_back( vSamples.size() );
					vSamples.insert(vSamples.end(), vvDataSet_All[dt][ca].second.begin() + vvDataSet_InstBegins_All[dt][ca].second[in],  vvDataSet_All[dt][ca].second.begin() + vvDataSet_InstBegins_All[dt][ca].second[in+1] );
				}
			}
			vInstanceBegins.push_back( vSamples.size() );
			

			mapDataSet[superCat].insert(mapDataSet[superCat].end(), vSamples.begin(), vSamples.end() ) ;

			size_t lastBegin = 0;
			if(mapInstBegins.find(superCat) != mapInstBegins.end())
			{
				lastBegin = mapInstBegins[superCat][ mapInstBegins[superCat].size()-1 ];
				mapInstBegins[superCat].resize(	mapInstBegins[superCat].size()-1 );
			}
			for(size_t in=0; in<vInstanceBegins.size(); in++)
			{
				vInstanceBegins[in] += lastBegin;
			}
			mapInstBegins[superCat].insert( mapInstBegins[superCat].end(), vInstanceBegins.begin(), vInstanceBegins.end() );


		}
	}

	vector< pair< string, vector<string> > > vDataSet_Full;
	DS_mapToVect(mapDataSet, vDataSet_Full);

	size_t nTotSamples = CountDatasetSamples(vDataSet_Full);

	//get 9/10 for training set or 1/10 for test set

	size_t nSamples_tot = 0;
	vDataSet.clear();
	m_vDataSet_InstBegins.clear();

	for(size_t ca=0; ca<vDataSet_Full.size() ; ca++)
	{
		string category = vDataSet_Full[ca].first;

		size_t nInstances = mapInstBegins[category].size()-1;
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
				vSamples.insert(vSamples.end(), vDataSet_Full[ca].second.begin() + mapInstBegins[category][in],  vDataSet_Full[ca].second.begin() + mapInstBegins[category][in+1] );
			}
		}
		vInstBegins.push_back(vSamples.size());

		vDataSet.push_back(pair< string, vector<string> > (category, vSamples));
		m_vDataSet_InstBegins.push_back( pair< string, vector<size_t> > (category, vInstBegins) ); 
		nSamples_tot += vSamples.size();
	}

	return nSamples_tot;
}
	


size_t HyperRGBD::HyperDataset::GetAllAbsFilenames(std::vector< std::string > &vAbsFilenames)
{
	const int nMaxCategories = (m_nMaxCategories_test == numeric_limits<int>::max())?m_nMaxCategories_train:max(m_nMaxCategories_train, m_nMaxCategories_test);
	const int nMaxInstances = (m_nMaxInstances_test == numeric_limits<int>::max())?m_nMaxInstances_train:max(m_nMaxInstances_train, m_nMaxInstances_test);
	const int nMaxInstanceViews = (m_nMaxInstanceViews_test == numeric_limits<int>::max())?m_nMaxInstanceViews_train:max(m_nMaxInstanceViews_train, m_nMaxInstanceViews_test);
	return GetAllAbsFilenames(nMaxCategories, nMaxInstances, nMaxInstanceViews, vAbsFilenames );
}


size_t HyperRGBD::HyperDataset::GetAllAbsFilenames(const int nMaxCategories, const int nMaxInstances, const int nMaxInstanceViews, std::vector< std::string > &vAbsFilenames)
{
	vector<string> vSamples;

	for(size_t dt=0; dt<m_datasets.size(); dt++)
	{

		m_datasets[dt].second->GetAllAbsFilenames(vSamples);
		vAbsFilenames.insert(vAbsFilenames.end(), vSamples.begin(), vSamples.end() );
	}

	return vAbsFilenames.size();
}



cv::Mat HyperRGBD::HyperDataset::ReadImage(const std::string &absFilename_withoutSuffix, const ImageType imageType)
{
	string tempRoot;
	for(size_t dt=0; dt<m_datasets.size(); dt++)
	{
		tempRoot=m_datasets[dt].second->GetRoot();
		size_t keyWrPos = absFilename_withoutSuffix.rfind(tempRoot);
		if (keyWrPos != string::npos)
		{
			return m_datasets[dt].second->ReadImage(absFilename_withoutSuffix, imageType);
		}
	}
	throw runtime_error("ERROR: the image does not belong to any known Dataset"); 
}


void HyperRGBD::HyperDataset::SetTrial(const int trial)
{
	m_trial=trial;
	for(size_t dt=0; dt<m_datasets.size(); dt++)
	{
		m_datasets[dt].second->SetTrial(trial);
	}
}

