#ifndef HYPERRGBD_CIN_2D3D_H
#define HYPERRGBD_CIN_2D3D_H

#include "hyperrgbdDefines.h"

#include "hyperrgbdIDataset.h"

#include <string>
#include <vector>
#include <map>
#include <set>


namespace HyperRGBD
{

    /** \brief Wrapper of the dataset introduced in:
    *  Browatzki B., Fischer J, Graf B., Bülthoff H.H, Wallraven C. "Going into depth: Evaluating 2D and 3D cues for object classification on a new, large-scale object dataset" 1st IEEE Workshop on Consumer Depth Cameras for Computer Vision.
    *
    * \author Alioscia Petrelli
    */
    class HYPER_RGBD_API CIN2D3D : virtual public IDataset 
    {
    public:

		/** \brief Number of categories in training set. For debug purpose.*/
        int m_nMaxCategories_train;

		/** \brief Number of instances in training set. For debug purpose.*/
        int	m_nMaxInstances_train;

		/** \brief Number of views in training set for each instance. For debug purpose.*/
        int	m_nMaxInstanceViews_train;

		/** \brief Number of categories in test set. For debug purpose.*/
        int m_nMaxCategories_test;

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
        CIN2D3D();

		/** \brief Constructor that sets dataset root*/
        CIN2D3D(const std::string &absDatasetRoot);

		/** \brief Initalization*/
        virtual void Init();


        size_t GetTrainingSet(const std::string &evalType, const int trial, std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames);
        size_t GetTestSet(const std::string &evalType, const int trial, std::vector< std::pair< std::string, std::vector<std::string> > > &vTestSet, const bool getAbsNames);

		/** \brief Splitting for category recognition as described in:
		*  A. Petrelli, L. Di Stefano "RGB-D Visual Search with Compact Binary Codes" 3DV, 2015
		*/
        size_t getDataSet_CatRec(const int trial, std::vector< std::pair< std::string, std::vector<std::string> > > &vDataSet, const bool getAbsNames, const bool isTrainingSet, const int nMaxCategories, const int nMaxInstances, const int nMaxInstanceViews);

		/** \brief Splitting for category recognition as described in:
		*  A. Petrelli, L. Di Stefano "RGB-D Visual Search with Compact Binary Codes" 3DV, 2015
		*/
        size_t getTrainingSet_CatRec(const int trial, std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames);

		/** \brief Splitting for category recognition as described in:
		*  A. Petrelli, L. Di Stefano "RGB-D Visual Search with Compact Binary Codes" 3DV, 2015
		*/
        size_t getTestSet_CatRec(const int trial, std::vector< std::pair< std::string, std::vector<std::string> > > &vTestSet, const bool getAbsNames);

		/** \brief Splitting for instance recognition as described in:
		*  A. Petrelli, L. Di Stefano "RGB-D Visual Search with Compact Binary Codes" 3DV, 2015
		*/
        size_t getDataSet_InstRec(const int trial, std::vector< std::pair< std::string, std::vector<std::string> > > &vDataSet, const bool getAbsNames, const bool isTrainingSet, const int nMaxCategories, const int nMaxInstances, const int nMaxInstanceViews);

		/** \brief Splitting for instance recognition as described in:
		*  A. Petrelli, L. Di Stefano "RGB-D Visual Search with Compact Binary Codes" 3DV, 2015
		*/
        size_t getTrainingSet_InstRec(const int trial, std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames);

		/** \brief Splitting for instance recognition as described in:
		*  A. Petrelli, L. Di Stefano "RGB-D Visual Search with Compact Binary Codes" 3DV, 2015
		*/
        size_t getTestSet_InstRec(const int trial, std::vector< std::pair< std::string, std::vector<std::string> > > &vTestSet, const bool getAbsNames);

		/** \brief Return all the dataset partitioned in categories.
		*
		* Used by HyperDataset class.
		*/
        size_t GetTrainingSet_CatRec_Full(std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames);

		/** \brief Return all the dataset partitioned in instances.
		*
		* Used by HyperDataset class.
		*/
        size_t GetTrainingSet_InstRec_Full(std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames);

        
        size_t GetAllAbsFilenames(const int nMaxCategories, const int nMaxInstances, const int nMaxInstanceViews, std::vector< std::string > &vAbsFilenames);


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
        static cv::Mat ReadDepthMap(const std::string &absFilename);

         /** \brief Return a range map (3-channel image (x,y,z) of float data)
         * \param[in] convertToMillimeters false measures are in 100um
         */
        static cv::Mat ReadRangeMap(const std::string &absFilename);

         /** \brief Return a mask (1-channel image of unsigned chars)
         */
        static cv::Mat ReadMask(const std::string &absFilename_withoutSuffix);


        virtual cv::Mat ReadImage(const std::string &absFilename_withoutSuffix, const ImageType imageType);

        virtual size_t GetTrainingSet(std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames);
        virtual size_t GetTestSet(std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames);

        virtual size_t GetAllAbsFilenames(std::vector< std::string > &vAbsFilenames);

        virtual std::string GetRoot(){return m_absDatasetRoot;}

        virtual int GetNumTrials(){return 10;};

        using IDataset::SetTrial;
        using IDataset::GetTrial;

    };
}

#endif