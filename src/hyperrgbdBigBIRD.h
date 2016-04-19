#ifndef HYPERRGBD_BIG_BIRD_H
#define HYPERRGBD_BIG_BIRD_H

#include "hyperrgbdDefines.h"

#include "hyperrgbdIDataset.h"

#include <string>
#include <vector>
#include <map>
#include <set>

#include "opencv2/opencv.hpp"


namespace HyperRGBD
{


    /** \brief Wrapper of the dataset introduced in:
    *  Arjun Singh, James Sha, Karthik S. Narayan, Tudor Achim, Pieter Abbeel "A Large-Scale 3D Database of Object Instances." ICRA 2014
    *
    * \author Alioscia Petrelli
    */
    class HYPER_RGBD_API BigBIRD : virtual public IDataset 
    {
    public:
        /** \brief Number of instances in training set. For debug purpose.*/
        int	m_nMaxInstances_train;

        /** \brief Number of views in training set for each instance. For debug purpose.*/
        int	m_nMaxInstanceViews_train;

        /** \brief Number of instances in test set. For debug purpose.*/
        int	m_nMaxInstances_test;

        /** \brief Number of views in test set for each instance. For debug purpose.*/
        int	m_nMaxInstanceViews_test;

        /** \brief Ratio for the splitting of the views in training and test sets.*/
        float		m_testSetRatio;

        /** \brief Training set as vector of pairs <label, vector of filenames>.*/
        std::vector< std::pair< std::string, std::vector<std::string> > > m_vTrainingSet;

        /** \brief Test set as vector of pairs <label, vector of filenames>.*/
        std::vector< std::pair< std::string, std::vector<std::string> > > m_vTestSet;

        /** \brief Empty constructor */
        BigBIRD();

        /** \brief Constructor that sets dataset root*/
        BigBIRD(const std::string &absDatasetRoot);

        /** \brief Initalization*/
        virtual void Init();

        size_t getDataSet(const std::string &evalType, const int trial, std::vector< std::pair< std::string, std::vector<std::string> > > &vDataSet, const bool getAbsNames, const bool isTrainingSet, const int nMaxInstances, const int nMaxInstanceViews);
        size_t GetTrainingSet(const std::string &evalType, const int trial, std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames);
        size_t GetTestSet(const std::string &evalType, const int trial, std::vector< std::pair< std::string, std::vector<std::string> > > &vTestSet, const bool getAbsNames);

        size_t GetAllAbsFilenames(const int nMaxInstances, const int nMaxInstanceViews, std::vector< std::string > &vAbsFilenames);

        /** \brief Given the absolute path denoting an acquisition, return the absolute filename of a specific format of image
        * \param[in] absFilename_withoutSuffix absolute path denoting an acquisition
        * \param[in] imageType image format
        * \return absolute filename of a specific format of image
        */
        static std::string GetImageFilename_AbsFull(const std::string &absFilename_withoutSuffix, const ImageType imageType);

        /** \brief Given a specific image format, return all the absolute filenames of images in that specific format
        * \param[out] vDataSet all the absolute filenames of images in a specific format
        * \param[in] imageType image format
        */
        static void GetImageFilename_AbsFull(std::vector< std::pair< std::string, std::vector<std::string> > > &vDataSet, const ImageType imageType);

         /** \brief Return a depth map (1-channel image of short data)
         * \param[in] convertToMillimeters false measures are in 100um
         */
        static cv::Mat ReadDepthMap(const std::string &absFilename, bool convertToMillimeters = false);

         /** \brief Return a range map (3-channel image (x,y,z) of float data)
         * \param[in] convertToMillimeters false measures are in 100um
         */
        static cv::Mat ReadRangeMap(const std::string &absFilename_withoutSuffix, bool convertToMillimeters = false);

        /** \brief Return calibration data. Used by ReadRangeMap */
        static cv::Mat ReadCalibrationData(const std::string &absFilename, const std::string &datasetName);

        virtual cv::Mat ReadImage(const std::string &absFilename_withoutSuffix, const ImageType imageType);

        virtual size_t GetTrainingSet(std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames);
        virtual size_t GetTestSet(std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames);

        virtual size_t GetAllAbsFilenames(std::vector< std::string > &vAbsFilenames);

        virtual int GetNumTrials(){return 10;};

        /** \brief Crop all the images of the dataset as described in:
        *    A. Petrelli, L. Di Stefano "RGB-D Visual Search with Compact Binary Codes" 3DV, 2015
        */
        void CropAllDataset(const std::string &absDatasetRoot, const std::string &absCroppedDatasetRoot, const float bBoxExpansionFactor = 1.5f, const int first = 0, const int last = std::numeric_limits<int>::max()); 

        /** \brief Convert all the images as performed by CropAllDataset without performing the cropping*/
        void AdaptAllDataset(const std::string &absDatasetRoot, const std::string &absAdaptedDatasetRoot, const int first = 0, const int last = std::numeric_limits<int>::max());

        using IDataset::SetTrial;
        using IDataset::GetTrial;
    };


    /** \brief Wrapper of the cropped version of BigBIRD dataset, as explained in:
    *  A. Petrelli, L. Di Stefano "RGB-D Visual Search with Compact Binary Codes" 3DV, 2015
    *
    * Run CropAllDataset method of BigBIRD for creating the dataset
    * \author Alioscia Petrelli
    */
    class HYPER_RGBD_API CroppedBIRD : virtual public BigBIRD 
    {
    public:
        float		m_allViewsRatio;


        CroppedBIRD();
        CroppedBIRD(const std::string &absDatasetRoot);

        virtual void Init();


        size_t GetTrainingSet(const std::string &evalType, const int trial, std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames);
        size_t GetTestSet(const std::string &evalType, const int trial, std::vector< std::pair< std::string, std::vector<std::string> > > &vTestSet, const bool getAbsNames);


        size_t getDataSet_AllInst(const int trial, std::vector< std::pair< std::string, std::vector<std::string> > > &vDataSet, const bool getAbsNames, const bool isTrainingSet, const int nMaxInstances, const int nMaxInstanceViews);
        virtual size_t getTrainingSet_AllInst(const int trial, std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames);
        virtual size_t getTestSet_AllInst(const int trial, std::vector< std::pair< std::string, std::vector<std::string> > > &vTestSet, const bool getAbsNames);

        size_t getDataSet_Full(std::vector< std::pair< std::string, std::vector<std::string> > > &vDataSet, const bool getAbsNames);

        virtual size_t GetAllAbsFilenames(const int nMaxInstances, const int nMaxInstanceViews, std::vector< std::string > &vAbsFilenames);


        static std::string GetImageFilename_AbsFull(const std::string &absFilename_withoutSuffix, const ImageType imageType);
        static void GetImageFilename_AbsFull(std::vector< std::pair< std::string, std::vector<std::string> > > &vDataSet, const ImageType imageType);


        virtual cv::Mat ReadImage(const std::string &absFilename_withoutSuffix, const ImageType imageType);

        virtual size_t GetTrainingSet(std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames);
        virtual size_t GetTestSet(std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames);

        virtual size_t GetAllAbsFilenames(std::vector< std::string > &vAbsFilenames);

        virtual int GetNumTrials(){return (m_evalType=="Full")?1:10;};

        bool DiscardObject(const std::string &objectName);

        virtual void CheckDataSetVisually(const std::string &absPath, const int first = 0);
        virtual void CopyDataSetForCheck(const std::string &absOutPath);

        using IDataset::SetTrial;
        using IDataset::GetTrial;

    };

}

#endif
