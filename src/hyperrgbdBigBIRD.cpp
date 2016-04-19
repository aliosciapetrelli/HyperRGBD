#include "hyperrgbdBigBIRD.h"
#include "hyperrgbdUtils.h"

#ifndef HDF5_IS_NOT_INCLUDED
#include "H5Cpp.h"
#endif


using namespace cv;
using namespace std;


void HyperRGBD::BigBIRD::Init()
{
	m_nMaxInstances_train = numeric_limits<int>::max();
	m_nMaxInstanceViews_train = numeric_limits<int>::max();

	m_nMaxInstances_test = numeric_limits<int>::max();
	m_nMaxInstanceViews_test = numeric_limits<int>::max();

	m_trial = -1;

	m_testSetRatio = 1/3.0f;

	m_evalType = "";
}

HyperRGBD::BigBIRD::BigBIRD(const string &absDatasetRoot)
{
	Init();

	SetRoot(absDatasetRoot);
}

HyperRGBD::BigBIRD::BigBIRD()
{
	Init();
}


cv::Mat HyperRGBD::BigBIRD::ReadDepthMap(const std::string &absFilename, bool convertToMillimeters)
{
	cv::Mat depthMap;

	#ifdef HDF5_IS_NOT_INCLUDED	
		throw runtime_error("ERROR: Include HDF5 library!!!!");
	#else

		H5::H5File file( absFilename.c_str(), H5F_ACC_RDONLY );
		H5::DataSet dataset = file.openDataSet("depth");

		H5::DataSpace filespace = dataset.getSpace();
		int rank = filespace.getSimpleExtentNdims();

		hsize_t dims[2]; 
		rank = filespace.getSimpleExtentDims( dims );


		//convert to Mat
		depthMap = Mat((int)dims[0], (int)dims[1], CV_16U);
		if(depthMap.isContinuous())
		{
			dataset.read( &depthMap.data[0], H5::PredType::NATIVE_USHORT, filespace, filespace );
		}
		else
		{
			vector<unsigned short> data_out(dims[0] * dims[1]);
			dataset.read( &data_out[0], H5::PredType::NATIVE_USHORT, filespace, filespace );

			for (int ro = 0; ro < (int)dims[0]; ro++)
			{
				memcpy(depthMap.row(ro).data, &data_out[ro*(int)dims[1]], sizeof(unsigned short)*(int)dims[1]);
			}
		}

		if(convertToMillimeters)
		{
			depthMap /= 10;
		}

		return depthMap;

	#endif
}

cv::Mat HyperRGBD::BigBIRD::ReadRangeMap(const std::string &absFilename_withoutSuffix, bool convertToMillimeters)
{
	Mat depthMap = ReadDepthMap(GetImageFilename_AbsFull(absFilename_withoutSuffix, IMAGETYPE_DEPTHMAP), false);

	string absPath = GetPathDir(absFilename_withoutSuffix);
	string absCalibrationFile = absPath + "/calibration.h5";

	string strCameraName = GetRelativeName(absFilename_withoutSuffix).substr(0, 3);
	string absIntrinsicMatrix = strCameraName + "_depth_K";

	Mat intrisicMatrix = ReadCalibrationData(absCalibrationFile, absIntrinsicMatrix);

	Mat intrisicMatrix_inverse;
	invert(intrisicMatrix, intrisicMatrix_inverse);

	Mat rangeMap = Mat(depthMap.rows, depthMap.cols, CV_32FC3);

	Mat depthPoint(3, 1, CV_32F); 
	depthPoint.at<float>(2,0) = 1.0f;
	for(int ro=0; ro<depthMap.rows; ro++)
	{
		for(int co=0; co<depthMap.cols; co++)
		{
			if( depthMap.at<unsigned short>(ro, co) == 0)
			{
				rangeMap.at<Vec3f>(ro, co) = 0.0f;
			}
			else
			{
				depthPoint.at<float>(0,0) = (float)co;
				depthPoint.at<float>(1,0) = (float)ro;
				
				Mat point3D(rangeMap.at<Vec3f>(ro, co), false);
				
				point3D = intrisicMatrix_inverse * depthPoint;
				//rangeMap.at<Vec3f>(ro, co)[0] = point3D.at<double>(0,0);
				//rangeMap.at<Vec3f>(ro, co)[1] = point3D.at<double>(1,0);
				//rangeMap.at<Vec3f>(ro, co)[2] = point3D.at<double>(2,0);
				//rangeMap.at<Vec3f>(ro, co) = intrisicMatrix_inverse * depthPoint;
				rangeMap.at<Vec3f>(ro, co) *= depthMap.at<unsigned short>(ro, co);
			}
		}
	}

	if(convertToMillimeters)
	{
		rangeMap /= 10.0f;
	}

	return rangeMap;
}


cv::Mat HyperRGBD::BigBIRD::ReadCalibrationData(const std::string &absFilename, const std::string &datasetName)
{
	cv::Mat datasetMatrix;

	#ifdef HDF5_IS_NOT_INCLUDED	
		throw runtime_error("ERROR: Include HDF5 library!!!!");
	#else

		H5::H5File file( absFilename.c_str(), H5F_ACC_RDONLY );
		H5::DataSet dataset = file.openDataSet(datasetName.c_str());

		H5::DataSpace filespace = dataset.getSpace();
		int rank = filespace.getSimpleExtentNdims();

		hsize_t dims[2]; 
		rank = filespace.getSimpleExtentDims( dims );


		datasetMatrix = Mat((int)dims[0], (int)dims[1], CV_64F);

		if(datasetMatrix.isContinuous())
		{
			dataset.read( &datasetMatrix.data[0], H5::PredType::NATIVE_DOUBLE, filespace, filespace );
		}
		else
		{
			vector<double> data_out(dims[0] * dims[1]);
			dataset.read( &data_out[0], H5::PredType::NATIVE_DOUBLE, filespace, filespace );

			for (int ro = 0; ro < (int)dims[0]; ro++)
			{
				memcpy(datasetMatrix.row(ro).data, &data_out[ro*(int)dims[1]], sizeof(double)*(int)dims[1]);
			}
		}

		//convert to float
		datasetMatrix.convertTo(datasetMatrix, CV_32FC3);

		return datasetMatrix;

	#endif
}

size_t HyperRGBD::BigBIRD::GetTrainingSet(const string &evalType, const int trial, std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames)
{
	return getDataSet(m_evalType, trial, vTrainingSet, getAbsNames, true, m_nMaxInstances_train, m_nMaxInstanceViews_train);
}

size_t HyperRGBD::BigBIRD::GetTestSet(const string &evalType, const int trial, std::vector< std::pair< std::string, std::vector<std::string> > > &vTestSet, const bool getAbsNames)
{
	return getDataSet(m_evalType, trial, vTestSet, getAbsNames, false, m_nMaxInstances_test, m_nMaxInstanceViews_test);
}


size_t HyperRGBD::BigBIRD::getDataSet(const string &evalType, const int trial, std::vector< std::pair< std::string, std::vector<std::string> > > &vDataSet, const bool getAbsNames, const bool isTrainingSet, const int nMaxInstances, const int nMaxInstanceViews)
{
	if(trial < 0 || trial > GetNumTrials())
	{
		cout << "ERROR (BigBIRD::getDataSet: (trial < 0 || trial > m_nTrials)   " << trial;
		getchar();
		exit(-1);
	}

	if(isTrainingSet)
	{
		cout << "Get Training Set" << endl;
	}
	else
	{
		cout << "Get Test Set" << endl;
	}

	size_t nSamples_tot = 0;
	vDataSet.clear();

	srand(trial);

	vector<string> vInstances_rel;
	FindDirsEndWith(m_absDatasetRoot, "", vInstances_rel, true, nMaxInstances);

	for(size_t in=0; in<vInstances_rel.size(); in++)
	{
		cout << "Scan and collect filenames. Cat: " << in << "/" << vInstances_rel.size() << "\r";

		vector<string> vSamples;
		for(int vi=1; vi<=5; vi++)
		{
			vector<bool> vTestSampleIds(120, false);
			size_t nTestSamples = Sample_random(vTestSampleIds, m_testSetRatio);

			for(size_t sa=0; sa<vTestSampleIds.size(); sa++)
			{
				if( (isTrainingSet && !vTestSampleIds[sa]) || (!isTrainingSet && vTestSampleIds[sa]) )
				{
					stringstream ssFilename;
					ssFilename << "NP" << vi << "_" << sa*3;
					vSamples.push_back(ssFilename.str());
				}
			}
		}

		string instancePath = vInstances_rel[in];
		if(getAbsNames)
		{
			Prefix_Path(instancePath, m_absDatasetRoot);
		}
		Prefix_Path(vSamples, instancePath);

		vDataSet.push_back( pair< string, vector<string> > (vInstances_rel[in], vSamples) );
		nSamples_tot += vSamples.size();
	}

	cout << endl;

	return nSamples_tot;
}


string HyperRGBD::BigBIRD::GetImageFilename_AbsFull(const string &absFilename_withoutSuffix, const ImageType imageType)
{
	switch(imageType)
	{
	case IMAGETYPE_RGB:
		return absFilename_withoutSuffix + ".jpg"; 
		break;
	case IMAGETYPE_DEPTHMAP:
		return absFilename_withoutSuffix + ".h5";
		break;
	case IMAGETYPE_RANGEMAP:
		//return absFilename_withoutSuffix + ".h5";
		break;
	case IMAGETYPE_MASK:
		return GetPathDir(absFilename_withoutSuffix) + "/masks/" + GetRelativeName(absFilename_withoutSuffix) + "_mask.pbm";
		break;
	case IMAGETYPE_POSE:
		return GetPathDir(absFilename_withoutSuffix) + "/poses/" + GetRelativeName(absFilename_withoutSuffix) + "_pose.h5";
		break;
	default:
		throw runtime_error("ERROR (BigBIRD::GetImageFilename_AbsFull): imageType does not exist");
	}

	return "";
}

void HyperRGBD::BigBIRD::GetImageFilename_AbsFull(std::vector< std::pair< std::string, std::vector<std::string> > > &vDataSet, const ImageType imageType)
{
	for(size_t cl=0; cl<vDataSet.size(); cl++)
	{
		for(size_t im=0; im<vDataSet[cl].second.size(); im++)
		{
			vDataSet[cl].second[im] = BigBIRD::GetImageFilename_AbsFull(vDataSet[cl].second[im], imageType);
		}
	}
}


size_t HyperRGBD::BigBIRD::GetAllAbsFilenames(const int nMaxInstances, const int nMaxInstanceViews, std::vector< std::string > &vAbsFilenames )
{
	vAbsFilenames.clear();

	vector<string> vInstanceDirs_rel;
	FindDirsEndWith(m_absDatasetRoot, "", vInstanceDirs_rel, true, nMaxInstances);

	for(size_t in=0; in<vInstanceDirs_rel.size(); in++)
	{
		cout << "Scan and collect filenames. Cat: " << in << "/" << vInstanceDirs_rel.size() << "\r";

		vector<string> vSamples(120);
		for(int vi=1; vi<=5; vi++)
		{
			for(int sa=0; sa<120; sa++)
			{
				stringstream ssFilename;
				ssFilename << "NP" << vi << "_" << sa*3;
				vSamples[vi*120 + sa] = ssFilename.str();
			}
		}

		Prefix_Path(vSamples, vInstanceDirs_rel[in]);
		vAbsFilenames.insert(vAbsFilenames.end(), vSamples.begin(), vSamples.end());
	}

	Prefix_Path(vAbsFilenames, m_absDatasetRoot);

	cout << endl;
	return vAbsFilenames.size();
}

cv::Mat HyperRGBD::BigBIRD::ReadImage(const std::string &absFilename_withoutSuffix, const ImageType imageType)
{
	switch(imageType)
	{
	case IMAGETYPE_RGB:
		return imread(GetImageFilename_AbsFull(absFilename_withoutSuffix, imageType)); 
		break;
	case IMAGETYPE_DEPTHMAP:
		return ReadDepthMap(GetImageFilename_AbsFull(absFilename_withoutSuffix, imageType), true); 
		break;
	case IMAGETYPE_RANGEMAP:
		return ReadRangeMap(absFilename_withoutSuffix, true);
		break;
	case IMAGETYPE_MASK:
		return 255 - imread(GetImageFilename_AbsFull(absFilename_withoutSuffix, imageType), -1);
		break;
	case IMAGETYPE_POSE:
		throw runtime_error("BigBIRD::ReadImage imageType pose to implement");
		break;
	default:
		throw runtime_error("ERROR (BigBIRD::ReadImage): imageType does not exist");
	}

	return Mat();
}

size_t HyperRGBD::BigBIRD::GetTrainingSet(std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames)
{
	return GetTrainingSet(m_evalType, m_trial, vTrainingSet, getAbsNames); 
}

size_t HyperRGBD::BigBIRD::GetTestSet(std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames)
{
	return GetTestSet(m_evalType, m_trial, vTrainingSet, getAbsNames); 
}

size_t HyperRGBD::BigBIRD::GetAllAbsFilenames(std::vector< std::string > &vAbsFilenames)
{
	const int nMaxInstances = (m_nMaxInstances_test == numeric_limits<int>::max())?m_nMaxInstances_train:max(m_nMaxInstances_train, m_nMaxInstances_test);
	const int nMaxInstanceViews = (m_nMaxInstanceViews_test == numeric_limits<int>::max())?m_nMaxInstanceViews_train:max(m_nMaxInstanceViews_train, m_nMaxInstanceViews_test);
	return GetAllAbsFilenames(nMaxInstances, nMaxInstanceViews, vAbsFilenames);
}

void HyperRGBD::BigBIRD::CropAllDataset(const std::string &absDatasetRoot, const std::string &absCroppedDatasetRoot, const float bBoxExpansionFactor, const int first, const int last)
{
	
	vector<string> vDirs;
	if(last == std::numeric_limits<int>::max() )
	{
		HyperRGBD::FindDirsStartWith(absDatasetRoot, "", vDirs, false, last);
	}
	else
	{
		HyperRGBD::FindDirsStartWith(absDatasetRoot, "", vDirs, false, last + 1);
	}

	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
	compression_params.push_back(100);

	for(size_t di=first; di<vDirs.size(); di++)
	{
		string absDir_crop = absCroppedDatasetRoot + "/" + GetRelativeName(vDirs[di]);
		CreateFullPath(absDir_crop);

		string absCalibration = vDirs[di] + "/calibration.h5";

		for(int ca=1; ca<6; ca++)
		{
			string relFilenameTemplate = "NP" + ToString(ca);

			string strExtrinsic_rgb_from_NP5 = "H_" + relFilenameTemplate + "_from_NP5";
			Mat extrinsic_rgb_from_NP5 = ReadCalibrationData(absCalibration, strExtrinsic_rgb_from_NP5);
			string strExtrinsic_depth_from_NP5 = "H_" + relFilenameTemplate + "_ir_from_NP5";
			Mat extrinsic_depth_from_NP5 = ReadCalibrationData(absCalibration, strExtrinsic_depth_from_NP5);

			Mat extrinsic_depth_from_NP5_inverse;
			invert(extrinsic_depth_from_NP5, extrinsic_depth_from_NP5_inverse);

			string strIntrinsic_rgb = relFilenameTemplate + "_rgb_K";
			Mat intrinsic_rgb = ReadCalibrationData(absCalibration, strIntrinsic_rgb);

			for(int de=0; de<360; de+=3)
			{
				//check if all output files exist
				string absRGB_crop = absDir_crop + "/" + relFilenameTemplate + "_" + ToString(de) + "_rgb.jpg";
				string absMask_crop = absDir_crop + "/" + relFilenameTemplate + "_" + ToString(de) + "_mask.png";
				string absDepth_crop = absDir_crop + "/" + relFilenameTemplate + "_" + ToString(de) + "_depth.png";
                #ifdef SNAPPY_IS_NOT_INCLUDED
                    string absRange_crop = absDir_crop + "/" + relFilenameTemplate + "_" + ToString(de) + "_range.bin";
                #else
                	string absRange_crop = absDir_crop + "/" + relFilenameTemplate + "_" + ToString(de) + "_range.snappy";
                #endif

				cout << absRGB_crop;

				if( ExistsFile(absRGB_crop) && ExistsFile(absDepth_crop) &&  ExistsFile(absRange_crop) )
				{
					cout << "  -  skipped" << endl;
					continue;
				}

				
				string absFilenameTemplate = vDirs[di] + "/" + relFilenameTemplate + "_" + ToString(de);
				Mat image_rgb = ReadImage(absFilenameTemplate, HyperRGBD::IMAGETYPE_RGB);
				Mat image_mask = ReadImage(absFilenameTemplate, HyperRGBD::IMAGETYPE_MASK);
				Mat image_depth = ReadImage(absFilenameTemplate, HyperRGBD::IMAGETYPE_DEPTHMAP);
				Mat image_range = ReadImage(absFilenameTemplate, HyperRGBD::IMAGETYPE_RANGEMAP);	






				Mat depth_masked = Mat(image_depth.rows, image_depth.cols, CV_8U);
				depth_masked = 0;

				Mat image_range_inmeters;
				image_range.copyTo(image_range_inmeters);
				image_range_inmeters /= 1000.0f;

				for(int ro=0; ro<image_depth.rows; ro++)
				{
					for(int co=0; co<image_depth.cols; co++)
					{
						if( image_depth.at<unsigned short>(ro, co) == 0)
						{
							continue;
						}

						Mat point3D_depth(image_range_inmeters.at<Vec3f>(ro, co));
						point3D_depth.push_back(1.0f);

						Mat point3D_rgb =  extrinsic_rgb_from_NP5 * extrinsic_depth_from_NP5_inverse * point3D_depth;
						point3D_rgb /= point3D_rgb.at<float>(3);
						point3D_rgb.pop_back();

						Mat point_rgb = intrinsic_rgb * point3D_rgb;
						point_rgb /= point_rgb.at<float>(2);

						int ro_proj = (int)point_rgb.at<float>(1);
						int co_proj = (int)point_rgb.at<float>(0);
						if( (ro_proj < 0) || (ro_proj >= image_mask.rows) )
						{
							cout << " - skip (" << ro << ", " << co << ")";
						}
						if( (co_proj < 0) || (co_proj >= image_mask.cols) )
						{
							cout << " - skip (" << ro << ", " << co << ")";
						}

						if(image_mask.at<unsigned char>((int)point_rgb.at<float>(1), (int)point_rgb.at<float>(0)) == 255)
						{
							depth_masked.at<unsigned char>(ro, co) = 255;
						}

					}
				}

				Mat image_mask_copy;
				image_mask.copyTo(image_mask_copy);

				//crop depth map and range map
				
				Rect bbDepth;
				if(!FindMaskBoundingBox(depth_masked, bbDepth, 2.0f)) 
				{
					cout << "  -  No contour" << endl;
					continue;
				}
				//vector<Point> contour_depth;
				//MiUt::FindBiggestContour(depth_masked, contour_depth);
				//Rect bbDepth = boundingRect(contour_depth); 
				bbDepth = HyperRGBD::Expand(bbDepth, bBoxExpansionFactor, depth_masked.cols, depth_masked.rows);

				Mat image_depth_cropped;
				image_depth(bbDepth).copyTo(image_depth_cropped);
				Mat image_range_cropped;
				image_range(bbDepth).copyTo(image_range_cropped);

				//crop rgb and mask image
				Rect bbMask;
				if(!FindMaskBoundingBox(image_mask, bbMask, 2.0f)) 
				{
					cout << "  -  No contour" << endl;
					continue;
				}
				//vector<Point> contour_mask;
				//MiUt::FindBiggestContour(image_mask, contour_mask);
				//Rect bbMask = boundingRect(contour_mask); 
				bbMask = Expand(bbMask, bBoxExpansionFactor, image_mask.cols, image_mask.rows);

				Mat image_rgb_cropped;
				image_rgb(bbMask).copyTo(image_rgb_cropped);
				cv::resize(image_rgb_cropped, image_rgb_cropped, image_depth_cropped.size());

				Mat image_mask_cropped;
				image_mask_copy(bbMask).copyTo(image_mask_cropped);
				cv::resize(image_mask_cropped, image_mask_cropped, image_depth_cropped.size());

				//save images
				imwrite(absRGB_crop, image_rgb_cropped, compression_params);
				imwrite(absMask_crop, image_mask_cropped);
				imwrite(absDepth_crop, image_depth_cropped);
                #ifdef SNAPPY_IS_NOT_INCLUDED
                    Write_Mat(absRange_crop, image_range_cropped, STORAGEMODE_BINARY);
                #else
                    Write_Mat(absRange_crop, image_range_cropped, STORAGEMODE_SNAPPY);
                #endif
				

				//image_depth_cropped *= 30;

				//cv::imshow("croppedImage_depth", image_depth_cropped);
				//cv::imshow("croppedImage_rgb", image_rgb_cropped);
				//cv::imshow("croppedImage_mask", image_mask_cropped);

				//cv::waitKey(0);


				//cv::Mat rangeMap_8U;
				//ConvertTo8U(image_range_cropped, rangeMap_8U);

				//vector<Mat> vComps;
				//cv::split(rangeMap_8U, vComps);
				//cv::imshow("X", vComps[0]);
				//cv::imshow("Y", vComps[1]);
				//cv::imshow("Z", vComps[2]);
				//cv::waitKey(0);

				//cv::Mat image_range_cropped_reloaded;
				//Read_Mat(absRange_crop, image_range_cropped_reloaded, STORAGEMODE_SNAPPY);


				//cv::Mat rangeMap_8U;
				//ConvertTo8U(image_range_cropped_reloaded, rangeMap_8U);

				//vector<Mat> vComps;
				//cv::split(rangeMap_8U, vComps);
				//cv::imshow("X", vComps[0]);
				//cv::imshow("Y", vComps[1]);
				//cv::imshow("Z", vComps[2]);
				//cv::waitKey(0);



				cout << "  -  OK" << endl;
			}
		}

	}
}




void HyperRGBD::BigBIRD::AdaptAllDataset(const std::string &absDatasetRoot, const std::string &absAdaptedDatasetRoot, const int first, const int last)
{
	
	vector<string> vDirs;
	if(last == std::numeric_limits<int>::max() )
	{
		HyperRGBD::FindDirsStartWith(absDatasetRoot, "", vDirs, false, last);
	}
	else
	{
		HyperRGBD::FindDirsStartWith(absDatasetRoot, "", vDirs, false, last + 1);
	}

	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
	compression_params.push_back(100);

	for(size_t di=first; di<vDirs.size(); di++)
	{
		string absDir_crop = absAdaptedDatasetRoot + "/" + GetRelativeName(vDirs[di]);
		CreateFullPath(absDir_crop);

		string absCalibration = vDirs[di] + "/calibration.h5";

		for(int ca=1; ca<6; ca++)
		{
			string relFilenameTemplate = "NP" + ToString(ca);

			string strExtrinsic_rgb_from_NP5 = "H_" + relFilenameTemplate + "_from_NP5";
			Mat extrinsic_rgb_from_NP5 = ReadCalibrationData(absCalibration, strExtrinsic_rgb_from_NP5);
			string strExtrinsic_depth_from_NP5 = "H_" + relFilenameTemplate + "_ir_from_NP5";
			Mat extrinsic_depth_from_NP5 = ReadCalibrationData(absCalibration, strExtrinsic_depth_from_NP5);

			Mat extrinsic_depth_from_NP5_inverse;
			invert(extrinsic_depth_from_NP5, extrinsic_depth_from_NP5_inverse);

			string strIntrinsic_rgb = relFilenameTemplate + "_rgb_K";
			Mat intrinsic_rgb = ReadCalibrationData(absCalibration, strIntrinsic_rgb);

			for(int de=0; de<360; de+=3)
			{
				//check if all output files exist
				string absRGB_crop = absDir_crop + "/" + relFilenameTemplate + "_" + ToString(de) + "_rgb.jpg";
				string absMask_crop = absDir_crop + "/" + relFilenameTemplate + "_" + ToString(de) + "_mask.png";
				string absDepth_crop = absDir_crop + "/" + relFilenameTemplate + "_" + ToString(de) + "_depth.png";
                #ifdef SNAPPY_IS_NOT_INCLUDED
				    string absRange_crop = absDir_crop + "/" + relFilenameTemplate + "_" + ToString(de) + "_range.bin";
                #else
                    string absRange_crop = absDir_crop + "/" + relFilenameTemplate + "_" + ToString(de) + "_range.snappy";
                #endif

				cout << absRGB_crop;

				if( ExistsFile(absRGB_crop) && ExistsFile(absDepth_crop) &&  ExistsFile(absRange_crop) )
				{
					cout << "  -  skipped" << endl;
					continue;
				}

				
				string absFilenameTemplate = vDirs[di] + "/" + relFilenameTemplate + "_" + ToString(de);
				Mat image_rgb = ReadImage(absFilenameTemplate, HyperRGBD::IMAGETYPE_RGB);
				Mat image_mask = ReadImage(absFilenameTemplate, HyperRGBD::IMAGETYPE_MASK);
				Mat image_depth = ReadImage(absFilenameTemplate, HyperRGBD::IMAGETYPE_DEPTHMAP);
				Mat image_range = ReadImage(absFilenameTemplate, HyperRGBD::IMAGETYPE_RANGEMAP);	






				Mat depth_masked = Mat(image_depth.rows, image_depth.cols, CV_8U);
				depth_masked = 0;

				Mat image_range_inmeters;
				image_range.copyTo(image_range_inmeters);
				image_range_inmeters /= 1000.0f;

				for(int ro=0; ro<image_depth.rows; ro++)
				{
					for(int co=0; co<image_depth.cols; co++)
					{
						if( image_depth.at<unsigned short>(ro, co) == 0)
						{
							continue;
						}

						Mat point3D_depth(image_range_inmeters.at<Vec3f>(ro, co));
						point3D_depth.push_back(1.0f);

						Mat point3D_rgb =  extrinsic_rgb_from_NP5 * extrinsic_depth_from_NP5_inverse * point3D_depth;
						point3D_rgb /= point3D_rgb.at<float>(3);
						point3D_rgb.pop_back();

						Mat point_rgb = intrinsic_rgb * point3D_rgb;
						point_rgb /= point_rgb.at<float>(2);

						int ro_proj = (int)point_rgb.at<float>(1);
						int co_proj = (int)point_rgb.at<float>(0);
						if( (ro_proj < 0) || (ro_proj >= image_mask.rows) )
						{
							cout << " - skip (" << ro << ", " << co << ")";
						}
						if( (co_proj < 0) || (co_proj >= image_mask.cols) )
						{
							cout << " - skip (" << ro << ", " << co << ")";
						}

						if(image_mask.at<unsigned char>((int)point_rgb.at<float>(1), (int)point_rgb.at<float>(0)) == 255)
						{
							depth_masked.at<unsigned char>(ro, co) = 255;
						}

					}
				}

				//Mat image_mask_copy;
				//image_mask.copyTo(image_mask_copy);

				//crop depth map and range map
				
				//Rect bbDepth;
				//if(!FindMaskBoundingBox(depth_masked, bbDepth, 2.0f)) 
				//{
				//	cout << "  -  No contour" << endl;
				//	continue;
				//}
				//vector<Point> contour_depth;
				//MiUt::FindBiggestContour(depth_masked, contour_depth);
				//Rect bbDepth = boundingRect(contour_depth); 
				
				//bbDepth = MiUt::Expand(bbDepth, bBoxExpansionFactor, depth_masked.cols, depth_masked.rows);

				//Mat image_depth_cropped;
				//image_depth(bbDepth).copyTo(image_depth_cropped);
				//Mat image_range_cropped;
				//image_range(bbDepth).copyTo(image_range_cropped);

				//crop rgb and mask image
				//Rect bbMask;
				//if(!FindMaskBoundingBox(image_mask, bbMask, 2.0f)) 
				//{
				//	cout << "  -  No contour" << endl;
				//	continue;
				//}
				//vector<Point> contour_mask;
				//MiUt::FindBiggestContour(image_mask, contour_mask);
				//Rect bbMask = boundingRect(contour_mask); 
				
				//bbMask = Expand(bbMask, bBoxExpansionFactor, image_mask.cols, image_mask.rows);

				//Mat image_rgb_cropped;
				//image_rgb(bbMask).copyTo(image_rgb_cropped);
				//cv::resize(image_rgb_cropped, image_rgb_cropped, image_depth_cropped.size());

				//Mat image_mask_cropped;
				//image_mask_copy(bbMask).copyTo(image_mask_cropped);
				//cv::resize(image_mask_cropped, image_mask_cropped, image_depth_cropped.size());

				//save images
				std::ifstream  fileRGB_src(GetImageFilename_AbsFull(absFilenameTemplate, HyperRGBD::IMAGETYPE_RGB), std::ios::binary);
				std::ofstream  fileRGB_dst(absRGB_crop,   std::ios::binary);
				fileRGB_dst << fileRGB_src.rdbuf();



				//imwrite(absRGB_crop, image_rgb, compression_params);
				imwrite(absMask_crop, image_mask);
				imwrite(absDepth_crop, image_depth);
				//Write_Mat(absRange_crop, image_range, STORAGEMODE_SNAPPY);		//TODO: each range map is about 1 Mb. Too much!!!! 
				//Write_Mat(absRange_crop, image_range, STORAGEMODE_BINARY);		//TODO: each range map is about 3 Mb. Too much!!!!

				//image_depth_cropped *= 30;

				//cv::imshow("croppedImage_depth", image_depth_cropped);
				//cv::imshow("croppedImage_rgb", image_rgb_cropped);
				//cv::imshow("croppedImage_mask", image_mask_cropped);

				//cv::waitKey(0);


				//cv::Mat rangeMap_8U;
				//ConvertTo8U(image_range_cropped, rangeMap_8U);

				//vector<Mat> vComps;
				//cv::split(rangeMap_8U, vComps);
				//cv::imshow("X", vComps[0]);
				//cv::imshow("Y", vComps[1]);
				//cv::imshow("Z", vComps[2]);
				//cv::waitKey(0);

				//cv::Mat image_range_cropped_reloaded;
				//Read_Mat(absRange_crop, image_range_cropped_reloaded, STORAGEMODE_SNAPPY);


				//cv::Mat rangeMap_8U;
				//ConvertTo8U(image_range_cropped_reloaded, rangeMap_8U);

				//vector<Mat> vComps;
				//cv::split(rangeMap_8U, vComps);
				//cv::imshow("X", vComps[0]);
				//cv::imshow("Y", vComps[1]);
				//cv::imshow("Z", vComps[2]);
				//cv::waitKey(0);



				cout << "  -  OK" << endl;
			}
		}

	}
}



void HyperRGBD::CroppedBIRD::Init()
{
	m_nMaxInstances_train = numeric_limits<int>::max();
	m_nMaxInstanceViews_train = numeric_limits<int>::max();

	m_nMaxInstances_test = numeric_limits<int>::max();
	m_nMaxInstanceViews_test = numeric_limits<int>::max();

	m_trial = -1;

	m_evalType = "";

	m_testSetRatio = 1/float(GetNumTrials());

	m_allViewsRatio = 1/6.0f;

}

HyperRGBD::CroppedBIRD::CroppedBIRD(const string &absDatasetRoot)
{
	Init();

	SetRoot(absDatasetRoot);
}

HyperRGBD::CroppedBIRD::CroppedBIRD()
{
	Init();
}



bool HyperRGBD::CroppedBIRD::DiscardObject(const string &objectName)
{
	return ((objectName == "coca_cola_glass_bottle") ||
			(objectName == "crayola_yellow_green") ||
			(objectName == "expo_marker_red") ||
			(objectName == "hersheys_bar") ||
			(objectName == "listerine_green") ||
			(objectName == "palmolive_green") ||
			(objectName == "palmolive_orange") ||
			(objectName == "softsoap_clear") ||
			(objectName == "softsoap_green") ||
			(objectName == "softsoap_purple") ||
			(objectName == "windex"));
}


size_t HyperRGBD::CroppedBIRD::getDataSet_AllInst(const int trial, vector< pair< string, vector<string> > > &vDataSet, const bool getAbsNames, const bool isTrainingSet, const int nMaxInstances, const int nMaxInstanceViews)
{
	size_t nSamples_tot = 0;
	vDataSet.clear();

	string suffix = "_rgb.jpg";
	size_t suffixSize = suffix.size();


	
	vector<string> vInstanceDirs_rel;
	FindDirsStartWith(m_absDatasetRoot, "", vInstanceDirs_rel, true, nMaxInstances);
	
	if(vInstanceDirs_rel.size() == 0)
	{
		throw runtime_error("ERROR (CroppedBIRD::getDataSet_AllInst): no instances: " + ToString(m_absDatasetRoot) );
	}

	//collect all files in all instance dirs
	for(size_t in=0; in<vInstanceDirs_rel.size(); in++)
	{
		cout << "Scan and collect filenames. Inst: " << in << "/" << vInstanceDirs_rel.size() << "\r";

		if(DiscardObject(vInstanceDirs_rel[in]))
		{
			continue;
		}

		string instanceDir_abs = m_absDatasetRoot + "/" + vInstanceDirs_rel[in];

		vector<string> vAllFiles;
		FindFilesStartEndWith(instanceDir_abs, "", suffix, vAllFiles, !getAbsNames);
		
		//collect all views for this trial 
		vector<bool> vAllViewsIds;
		size_t nAllViews = 0;
		if(nMaxInstanceViews == numeric_limits<int>::max())
		{
			vAllViewsIds.resize(vAllFiles.size(), false);
			nAllViews = Sample_random(vAllViewsIds, m_allViewsRatio, trial);
		}
		else
		{
			vAllViewsIds.resize(nMaxInstanceViews, true);
			nAllViews = nMaxInstanceViews;
		}
		
		vector<string> vAllViews;
		vAllViews.reserve(nAllViews);
		for(size_t vi=0; vi<vAllViewsIds.size(); vi++)
		{
			if(vAllViewsIds[vi])
			{
				vAllViews.push_back(vAllFiles[vi]);
			}
		}

		//collect test views
		vector<bool> vTestSampleIds(vAllViews.size(), false);
		size_t nTestSamples = Sample_random(vTestSampleIds, m_testSetRatio, trial);

		if(nTestSamples == 0)
		{
			throw runtime_error("ERROR (CroppedBIRD::getDataSet_AllInst): (nTestSamples == 0) ");
		}

		vector<string> vSamples;
		for(size_t sa=0; sa<vTestSampleIds.size(); sa++)
		{
			if( (isTrainingSet && !vTestSampleIds[sa]) || (!isTrainingSet && vTestSampleIds[sa]) )
			{
				vAllViews[sa].resize(vAllViews[sa].size() - suffixSize);
				vSamples.push_back(vAllViews[sa]);
			}
		}

		if(nMaxInstanceViews != numeric_limits<int>::max())
		{
			vSamples.resize(min((size_t)nMaxInstanceViews, vSamples.size()));
		}

		//Prefix instance Dir
		if(!getAbsNames)
		{
			Prefix_Path(vSamples, vInstanceDirs_rel[in]);
		}


		vDataSet.push_back( pair< string, vector<string> > (vInstanceDirs_rel[in], vSamples) );
		nSamples_tot += vSamples.size();
	}


	cout << endl;

	return nSamples_tot;

}


size_t HyperRGBD::CroppedBIRD::getDataSet_Full(std::vector< std::pair< std::string, std::vector<std::string> > > &vDataSet, const bool getAbsNames)
{
	size_t nSamples_tot = 0;
	vDataSet.clear();

	string suffix = "_rgb.jpg";
	size_t suffixSize = suffix.size();


	
	vector<string> vInstanceDirs_rel;
	FindDirsStartWith(m_absDatasetRoot, "", vInstanceDirs_rel, true);
	
	if(vInstanceDirs_rel.size() == 0)
	{
		throw runtime_error("ERROR (CroppedBIRD::getDataSet_AllInst): no instances: " + ToString(m_absDatasetRoot) );
	}

	//collect all files in all instance dirs
	for(size_t in=0; in<vInstanceDirs_rel.size(); in++)
	{
		cout << "Scan and collect filenames. Inst: " << in << "/" << vInstanceDirs_rel.size() << "\r";

		if(DiscardObject(vInstanceDirs_rel[in]))
		{
			continue;
		}

		string instanceDir_abs = m_absDatasetRoot + "/" + vInstanceDirs_rel[in];

		vector<string> vAllFiles;
		FindFilesStartEndWith(instanceDir_abs, "", suffix, vAllFiles, !getAbsNames);
		
		vector<string> vSamples;
		for(size_t sa=0; sa<vAllFiles.size(); sa++)
		{
			vAllFiles[sa].resize(vAllFiles[sa].size() - suffixSize);
			vSamples.push_back(vAllFiles[sa]);
		}


		//Prefix instance Dir
		if(!getAbsNames)
		{
			Prefix_Path(vSamples, vInstanceDirs_rel[in]);
		}


		vDataSet.push_back( pair< string, vector<string> > (vInstanceDirs_rel[in], vSamples) );
		nSamples_tot += vSamples.size();
	}


	cout << endl;

	return nSamples_tot;

}



size_t HyperRGBD::CroppedBIRD::getTrainingSet_AllInst(const int trial, vector< pair< string, vector<string> > > &vTrainingSet, const bool getAbsNames)
{
	return getDataSet_AllInst(trial, vTrainingSet, getAbsNames, true, m_nMaxInstances_train, m_nMaxInstanceViews_train);
}



size_t HyperRGBD::CroppedBIRD::getTestSet_AllInst(const int trial, vector< pair< string, vector<string> > > &vTestSet, const bool getAbsNames)
{
	return getDataSet_AllInst(trial, vTestSet, getAbsNames, false, m_nMaxInstances_test, m_nMaxInstanceViews_test);
}


size_t HyperRGBD::CroppedBIRD::GetTrainingSet(const string &evalType, const int trial, vector< pair< string, vector<string> > > &vTrainingSet, const bool getAbsNames)
{
	if(m_absDatasetRoot == "")
	{
		cout << "ERROR (GetTrainingSet): m_absDatasetRoot not set";
	}

	if(evalType == "AllInst")
	{
		return getTrainingSet_AllInst(trial, vTrainingSet, getAbsNames);
	}
	else if( (evalType == "InstRec_Full") || (evalType == "Full") )
	{
		return getDataSet_Full(vTrainingSet, getAbsNames);
	}
	else
	{
		throw runtime_error("ERROR: Uncorrect evalType: " + evalType);
	}
}

size_t HyperRGBD::CroppedBIRD::GetTestSet(const string &evalType, const int trial, vector< pair< string, vector<string> > > &vTestSet, const bool getAbsNames)
{
	if(m_absDatasetRoot == "")
	{
		cout << "ERROR (GetTestSet): m_absDatasetRoot not set";
	}

	if(evalType == "AllInst")
	{
		return getTestSet_AllInst(trial, vTestSet, getAbsNames);
	}
	else
	{
		cout << "ERROR: Uncorrect evalType: " << evalType << endl;
		getchar();
		exit(-1);
	}
}


string HyperRGBD::CroppedBIRD::GetImageFilename_AbsFull(const string &absFilename_withoutSuffix, const ImageType imageType)
{
	switch(imageType)
	{
	case IMAGETYPE_AS_IS:
		return absFilename_withoutSuffix + "_rgb.jpg"; 
		break;
	case IMAGETYPE_RGB:
		return absFilename_withoutSuffix + "_rgb.jpg"; 
		break;
	case IMAGETYPE_DEPTHMAP:
		return absFilename_withoutSuffix + "_depth.png";
		break;
	case IMAGETYPE_RANGEMAP:
        #ifdef SNAPPY_IS_NOT_INCLUDED
		    return absFilename_withoutSuffix + "_range.bin";
        #else
            return absFilename_withoutSuffix + "_range.snappy";
        #endif

		break;
	case IMAGETYPE_MASK:
		return absFilename_withoutSuffix + "_mask.png";
		break;
	default:
		throw runtime_error("ERROR (CroppedBIRD::GetImageFilename_AbsFull): imageType does not exist");
	}

	return "";
}

void HyperRGBD::CroppedBIRD::GetImageFilename_AbsFull(std::vector< std::pair< std::string, std::vector<std::string> > > &vDataSet, const ImageType imageType)
{
	for(size_t cl=0; cl<vDataSet.size(); cl++)
	{
		for(size_t im=0; im<vDataSet[cl].second.size(); im++)
		{
			vDataSet[cl].second[im] = CroppedBIRD::GetImageFilename_AbsFull(vDataSet[cl].second[im], imageType);
		}
	}
}


size_t HyperRGBD::CroppedBIRD::GetAllAbsFilenames(const int nMaxInstances, const int nMaxInstanceViews, std::vector< std::string > &vAbsFilenames )
{
	vAbsFilenames.clear();

	string suffix = "_rgb.jpg";
	

	vector<string> vInstanceDirs_rel;
	FindDirsStartWith(m_absDatasetRoot, "", vInstanceDirs_rel, true, nMaxInstances);

	//collect all files in all instance dirs
	for(size_t in=0; in<vInstanceDirs_rel.size(); in++)
	{
		cout << "Scan and collect filenames. Inst: " << in << "/" << vInstanceDirs_rel.size() << "\r";

		if(DiscardObject(vInstanceDirs_rel[in]))
		{
			continue;
		}

		vector<string> vSamples;
		string instanceDir_abs = m_absDatasetRoot + "/" + vInstanceDirs_rel[in];
		FindFilesEndWith(instanceDir_abs, suffix, vSamples, false, nMaxInstanceViews);

		//if(vSamples.size() != 600)
		//{
		//	throw runtime_error("ERROR (CroppedBIRD::GetAllAbsFilenames): (vSamples.size() != 600) " + instanceDir_abs);
		//}

		vAbsFilenames.insert(vAbsFilenames.end(), vSamples.begin(), vSamples.end() );
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


cv::Mat HyperRGBD::CroppedBIRD::ReadImage(const std::string &absFilename_withoutSuffix, const ImageType imageType)
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
		return imread(GetImageFilename_AbsFull(absFilename_withoutSuffix, imageType), -1); 
		break;
	case IMAGETYPE_RANGEMAP:
		{
			cv::Mat rangeMap;
            #ifdef SNAPPY_IS_NOT_INCLUDED
			    Read_Mat(GetImageFilename_AbsFull(absFilename_withoutSuffix, imageType), rangeMap, STORAGEMODE_BINARY);
            #else
                Read_Mat(GetImageFilename_AbsFull(absFilename_withoutSuffix, imageType), rangeMap, STORAGEMODE_SNAPPY);
            #endif
			return rangeMap; 
			break;
		}
	case IMAGETYPE_MASK:
		return imread(GetImageFilename_AbsFull(absFilename_withoutSuffix, imageType), -1); 
		break;
	default:
		throw runtime_error("ERROR (CroppedBIRD::ReadImage): imageType does not exist");
	}

	return Mat();
}

size_t HyperRGBD::CroppedBIRD::GetTrainingSet(std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames)
{
	return GetTrainingSet(m_evalType, m_trial, vTrainingSet, getAbsNames); 
}

size_t HyperRGBD::CroppedBIRD::GetTestSet(std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames)
{
	return GetTestSet(m_evalType, m_trial, vTrainingSet, getAbsNames); 
}

size_t HyperRGBD::CroppedBIRD::GetAllAbsFilenames(std::vector< std::string > &vAbsFilenames)
{
	const int nMaxInstances = (m_nMaxInstances_test == numeric_limits<int>::max())?m_nMaxInstances_train:max(m_nMaxInstances_train, m_nMaxInstances_test);
	const int nMaxInstanceViews = (m_nMaxInstanceViews_test == numeric_limits<int>::max())?m_nMaxInstanceViews_train:max(m_nMaxInstanceViews_train, m_nMaxInstanceViews_test);
	return GetAllAbsFilenames(nMaxInstances, nMaxInstanceViews, vAbsFilenames );
}

void HyperRGBD::CroppedBIRD::CopyDataSetForCheck(const std::string &absOutPath)
{
	vector<string> vDirs;
	HyperRGBD::FindDirsStartWith(m_absDatasetRoot, "", vDirs);
	for(size_t di=0; di<vDirs.size(); di++)
	{
		cout << vDirs[di] << endl;

		string absOutFullPath = absOutPath + "/" + GetRelativeName(vDirs[di]);
		CreateFullPath(absOutFullPath);

		vector<string> vRGB;
		HyperRGBD::FindFilesEndWith(vDirs[di], "_rgb.jpg", vRGB);
		vector<string> vDepth;
		HyperRGBD::FindFilesEndWith(vDirs[di], "_depth.png", vDepth);
		vector<string> vMask;
		HyperRGBD::FindFilesEndWith(vDirs[di], "_mask.png", vMask);

		if( (vRGB.size() != vDepth.size()) || (vRGB.size() != vMask.size()) )
		{
			throw runtime_error(GetRelativeName(vDirs[di]));
		}

		for(size_t fi=0; fi<vRGB.size(); fi++)
		{
			Mat rgb = imread(vRGB[fi]);
			Mat depth = imread(vDepth[fi], -1);
			Mat mask = imread(vMask[fi], -1);

			depth *= 30;

			string absOut = absOutFullPath + "/" + GetRelativeName(vRGB[fi]);
			imwrite(absOut, rgb);
			absOut = absOutFullPath + "/" + GetRelativeName(vDepth[fi]);
			imwrite(absOut, depth);
			absOut = absOutFullPath + "/" + GetRelativeName(vMask[fi]);
			imwrite(absOut, mask);

			//cv::imshow("depth", depth);
			//cv::imshow("rgb", rgb);
			//cv::imshow("mask", mask);
			//cv::moveWindow("depth", 100, 100);
			//cv::moveWindow("rgb", 100 + depth.cols + 10, 100);
			//cv::moveWindow("mask", 100 + depth.cols + mask.cols + 20, 100);

			//cv::waitKey(0);
		}
	}
}





void HyperRGBD::CroppedBIRD::CheckDataSetVisually(const std::string &absPath, const int first)
{
	vector<string> vDirs;
	HyperRGBD::FindDirsStartWith(absPath, "", vDirs);
	for(size_t di=first; di<vDirs.size(); di++)
	{
		cout << di << " - " << vDirs[di] << endl;

		vector<string> vRGB;
		HyperRGBD::FindFilesEndWith(vDirs[di], "_rgb.jpg", vRGB);
		vector<string> vDepth;
		HyperRGBD::FindFilesEndWith(vDirs[di], "_depth.png", vDepth);
		vector<string> vMask;
		HyperRGBD::FindFilesEndWith(vDirs[di], "_mask.png", vMask);

		if( (vRGB.size() != vDepth.size()) || (vRGB.size() != vMask.size()) )
		{
			throw runtime_error(GetRelativeName(vDirs[di]));
		}

		for(size_t fi=0; fi<vRGB.size(); fi++)
		{
			Mat rgb = imread(vRGB[fi]);
			Mat depth = imread(vDepth[fi], -1);
			Mat mask = imread(vMask[fi], -1);

			//depth *= 30;

			//string absOut = absOutFullPath + "/" + GetRelativeName(vRGB[fi]);
			//imwrite(absOut, rgb);
			//absOut = absOutFullPath + "/" + GetRelativeName(vDepth[fi]);
			//imwrite(absOut, depth);
			//absOut = absOutFullPath + "/" + GetRelativeName(vMask[fi]);
			//imwrite(absOut, mask);

			cv::imshow("depth", depth);
			cv::imshow("rgb", rgb);
			cv::imshow("mask", mask);
			cv::moveWindow("depth", 100, 100);
			cv::moveWindow("rgb", 100 + depth.cols + 50, 100);
			cv::moveWindow("mask", 100 + depth.cols + mask.cols + 100, 100);

			char ret = cv::waitKey(0);
			if(ret == 'q')
			{
				break;
			}
		}
	}
}