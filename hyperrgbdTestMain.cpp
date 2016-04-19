#include "HyperRGBD.h"

#include <stdexcept>

using namespace std;

void Test_HyperRGBD_BigBIRD()
{
	HyperRGBD::BigBIRD bigBIRD;

	bigBIRD.CropAllDataset("D:/DEVELOP/Data/3D/BigBird", "D:/DEVELOP/Data/3D/CroppedBIRD", 1.5f); 
}

void Test_HyperRGBD_CroppedBIRD()
{
	HyperRGBD::CroppedBIRD croppedBIRD;

	croppedBIRD.SetRoot("D:/DEVELOP/Data/3D/CroppedBIRD");

	croppedBIRD.SetEvalType("AllInst");
	croppedBIRD.SetTrial(2);

	vector< pair< string, vector<string> > > vTrainingSet;
	vector< pair< string, vector<string> > > vTestSet;
	croppedBIRD.GetTrainingSet( vTrainingSet, true );
	croppedBIRD.GetTestSet( vTestSet, true );

	cv::Mat depthMap = croppedBIRD.ReadImage( vTrainingSet[2].second[5], HyperRGBD::IMAGETYPE_DEPTHMAP);

}

void Test_HyperRGBD_CIN2D3D()
{
	HyperRGBD::CIN2D3D cin2D3D;

	cin2D3D.SetRoot("D:/DEVELOP/Data/3D/CIN-2D_3D");

	cin2D3D.SetEvalType("InstRec");	//"CatRec" for category recognition
	cin2D3D.SetTrial(2);

	vector< pair< string, vector<string> > > vTrainingSet;
	vector< pair< string, vector<string> > > vTestSet;
	cin2D3D.GetTrainingSet( vTrainingSet, true );
	cin2D3D.GetTestSet( vTestSet, true );

	cv::Mat depthMap = cin2D3D.ReadImage( vTrainingSet[2].second[5], HyperRGBD::IMAGETYPE_DEPTHMAP);
}

void Test_HyperRGBD_MVRED()
{
	HyperRGBD::MVRED mvRED;

	mvRED.SetRoot("D:/DEVELOP/Data/3D/MV-Red");

	mvRED.CropDataset("D:/DEVELOP/Data/3D/CroppedMV-Red", false, 1.5f);
}

void Test_HyperRGBD_CroppedMVRED()
{
	HyperRGBD::CroppedMVRED croppedMVRED;

	croppedMVRED.SetRoot("D:/DEVELOP/Data/3D/CroppedMV-RED");

	croppedMVRED.SetEvalType("InstRec");	//"CatRec" for category recognition
	croppedMVRED.SetTrial(2);

	vector< pair< string, vector<string> > > vTrainingSet;
	vector< pair< string, vector<string> > > vTestSet;
	croppedMVRED.GetTrainingSet( vTrainingSet, true );
	croppedMVRED.GetTestSet( vTestSet, true );

	cv::Mat depthMap = croppedMVRED.ReadImage( vTrainingSet[2].second[5], HyperRGBD::IMAGETYPE_DEPTHMAP);
}

void Test_HyperRGBD_Washington()
{
	HyperRGBD::Washington washington;

	washington.SetRoot("D:/DEVELOP/Data/3D/BoRGBD");
	//or
	// washington.m_absEvalDir = "D:/DEVELOP/Data/3D/BoRGBD/rgbd-dataset_eval"
	// washington.m_absLeaveOneOutCategoryInstances = "D:/DEVELOP/Data/3D/BoRGBD/testinstance_ids.txt"

	
	washington.SetEvalType("CatRec");	//"InstRec_ACF" for "Alternating Contiguous Frames" instance recognition
										//"InstRec_LSO" for "Leave Sequence Out" instance recognition
	
	washington.SetTrial(2);

	vector< pair< string, vector<string> > > vTrainingSet;
	vector< pair< string, vector<string> > > vTestSet;
	washington.GetTrainingSet( vTrainingSet, true );
	washington.GetTestSet( vTestSet, true );

	cv::Mat depthMap = washington.ReadImage( vTrainingSet[2].second[5], HyperRGBD::IMAGETYPE_DEPTHMAP);
}

void Test_HyperRGBD_HyperDataset()
{
	HyperRGBD::HyperDataset hyperDataset;

	hyperDataset.SetRoot("D:/DEVELOP/Data/3D/BoRGBD");
	hyperDataset.m_datasets[0].second->SetRoot("D:/DEVELOP/Data/3D/BoRGBD");	//Washington
	hyperDataset.m_datasets[1].second->SetRoot("D:/DEVELOP/Data/3D/CIN-2D_3D");	//CIN2D3D
	hyperDataset.m_datasets[2].second->SetRoot("D:/DEVELOP/Data/3D/CroppedBIRD");	//CroppedBIRD
	hyperDataset.m_datasets[3].second->SetRoot("D:/DEVELOP/Data/3D/CroppedMV-RED");	//CroppedMVRED


	hyperDataset.SetEvalType("CatRec");	//"CatRec_Balanced"
										//"InstRec"
										//"InstRec_Balanced"
	
	hyperDataset.SetTrial(2);

	vector< pair< string, vector<string> > > vTrainingSet;
	vector< pair< string, vector<string> > > vTestSet;
	hyperDataset.GetTrainingSet( vTrainingSet, true );
	hyperDataset.GetTestSet( vTestSet, true );

	cv::Mat depthMap = hyperDataset.ReadImage( vTrainingSet[2].second[5], HyperRGBD::IMAGETYPE_DEPTHMAP);
}





int main()
{
	string test = "HyperRGBD_HyperDataset";

	if(test == "HyperRGBD_BigBIRD")
	{
		Test_HyperRGBD_BigBIRD();
	}
	else if(test == "HyperRGBD_CroppedBIRD")
	{
		Test_HyperRGBD_CroppedBIRD();
	}
	else if(test == "HyperRGBD_CIN2D3D")
	{
		Test_HyperRGBD_CIN2D3D();
	}
	else if(test == "HyperRGBD_MVRED")
	{
		Test_HyperRGBD_MVRED();
	}
	else if(test == "HyperRGBD_CroppedMVRED")
	{
		Test_HyperRGBD_CroppedMVRED();
	}
	else if(test == "HyperRGBD_Washington")
	{
		Test_HyperRGBD_Washington();
	}
	else if(test == "HyperRGBD_HyperDataset")
	{
		Test_HyperRGBD_HyperDataset();
	}
	else
	{
		throw runtime_error("ERROR: Uncorrect test: " + test);
	}

}

