#ifndef HYPERRGBD_HYPERDATASET
#define HYPERRGBD_HYPERDATASET

#include "hyperrgbdDefines.h"

#include "hyperrgbdIDataset.h"


#include "hyperrgbdCIN2D3D.h"
#include "hyperrgbdBigBIRD.h"
#include "hyperrgbdMVRED.h"
#include "hyperrgbdWashington.h"


namespace HyperRGBD
{
    /** \brief Wrapper of the dataset introduced in:
    *  A. Petrelli, L. Di Stefano "Learning to Weight Color And Depth for RGB-D Image Search" ECCV, 2016
    *
    * \author Alioscia Petrelli
    */
    class HYPER_RGBD_API HyperDataset : virtual public IDataset 
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

		/** \brief For each dataset, mapping between the categories of the dataset and the categories of the HyperDataset.
		*
		* map< dataset, map< category, HyperDataset_category > >. Used for category recognition
		*/
        std::map< std::string, std::map< std::string, std::string> > m_mapCatAssociations;

		/** \brief Vector of pairs <datasetName, Idatasets> */
		std::vector< std::pair < std::string,IDataset*> > m_datasets;

		/** \brief Empty constructor */
        HyperDataset();

		/** \brief Initalization*/
        virtual void Init();

		/** \brief Return the training set for each involved dataset. 
		*
		* \param[out] vNames	names of the datasets
		* \param[out] vvDataSets Each element of the vector represents the training set of a dataset
		*/		
        size_t GetTrainingSet_AllDatasets(const std::string &evalType, const bool getAbsNames, std::vector<std::string> &vNames, std::vector < std::vector < std::pair< std::string, std::vector< std::string> > > > &vvDataSets, std::vector < std::vector < std::pair< std::string, std::vector< size_t> > > > &vvDataSet_InstBegins);

		/** \brief Return the training set for each involved dataset. 
		*
		* \param[out] vNames	names of the datasets
		* \param[out] vvDataSets Each element of the vector represents the training set of a dataset
		*/
        size_t GetTrainingSet_AllDatasets(const std::string &evalType, const bool getAbsNames, std::vector<std::string> &vNames, std::vector < std::vector < std::pair< std::string, std::vector< std::string> > > > &vvDataSets);


        size_t GetTrainingSet(const std::string &evalType, const int trial, std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames);
        size_t GetTestSet(const std::string &evalType, const int trial, std::vector< std::pair< std::string, std::vector<std::string> > > &vTestSet, const bool getAbsNames);

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

		/** \brief Splitting for category recognition as described in:
		*  A. Petrelli, L. Di Stefano "Learning to Weight Color And Depth for RGB-D Image Search" ECCV, 2016
		*/
        size_t getDataSet_CatRec(const int trial, std::vector< std::pair< std::string, std::vector<std::string> > > &vDataSet, const bool getAbsNames, const bool isTrainingSet, const int nMaxCategories, const int nMaxInstances, const int nMaxInstanceViews);

		/** \brief Splitting for category recognition as described in:
		*  A. Petrelli, L. Di Stefano "Learning to Weight Color And Depth for RGB-D Image Search" ECCV, 2016
		*/
        size_t getTrainingSet_CatRec(const int trial, std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames);

		/** \brief Splitting for category recognition as described in:
		*  A. Petrelli, L. Di Stefano "Learning to Weight Color And Depth for RGB-D Image Search" ECCV, 2016
		*/
        size_t getTestSet_CatRec(const int trial, std::vector< std::pair< std::string, std::vector<std::string> > > &vTestSet, const bool getAbsNames);

		/** \brief Splitting for category recognition (balanced) as described in:
		*  A. Petrelli, L. Di Stefano "Learning to Weight Color And Depth for RGB-D Image Search" ECCV, 2016
		*/
        size_t getDataSet_CatRec_Balanced(const int trial, std::vector< std::pair< std::string, std::vector<std::string> > > &vDataSet, const bool getAbsNames, const bool isTrainingSet, const int nMaxCategories, const int nMaxInstances, const int nMaxInstanceViews);

		/** \brief Splitting for category recognition (balanced) as described in:
		*  A. Petrelli, L. Di Stefano "Learning to Weight Color And Depth for RGB-D Image Search" ECCV, 2016
		*/
        size_t getTrainingSet_CatRec_Balanced(const int trial, std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames);

		/** \brief Splitting for category recognition (balanced) as described in:
		*  A. Petrelli, L. Di Stefano "Learning to Weight Color And Depth for RGB-D Image Search" ECCV, 2016
		*/
        size_t getTestSet_CatRec_Balanced(const int trial, std::vector< std::pair< std::string, std::vector<std::string> > > &vTestSet, const bool getAbsNames);

		/** \brief Splitting for instance recognition as described in:
		*  A. Petrelli, L. Di Stefano "Learning to Weight Color And Depth for RGB-D Image Search" ECCV, 2016
		*/
        size_t getDataSet_InstRec(const int trial, std::vector< std::pair< std::string, std::vector<std::string> > > &vDataSet, const bool getAbsNames, const bool isTrainingSet, const int nMaxCategories, const int nMaxInstances, const int nMaxInstanceViews);

		/** \brief Splitting for instance recognition as described in:
		*  A. Petrelli, L. Di Stefano "Learning to Weight Color And Depth for RGB-D Image Search" ECCV, 2016
		*/
        size_t getTrainingSet_InstRec(const int trial, std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames);

		/** \brief Splitting for instance recognition as described in:
		*  A. Petrelli, L. Di Stefano "Learning to Weight Color And Depth for RGB-D Image Search" ECCV, 2016
		*/
        size_t getTestSet_InstRec(const int trial, std::vector< std::pair< std::string, std::vector<std::string> > > &vTestSet, const bool getAbsNames);

		/** \brief Splitting for instance recognition (balanced) as described in:
		*  A. Petrelli, L. Di Stefano "Learning to Weight Color And Depth for RGB-D Image Search" ECCV, 2016
		*/
        size_t getDataSet_InstRec_Balanced(const int trial, std::vector< std::pair< std::string, std::vector<std::string> > > &vDataSet, const bool getAbsNames, const bool isTrainingSet, const int nMaxCategories, const int nMaxInstances, const int nMaxInstanceViews);

		/** \brief Splitting for instance recognition (balanced) as described in:
		*  A. Petrelli, L. Di Stefano "Learning to Weight Color And Depth for RGB-D Image Search" ECCV, 2016
		*/
        size_t getTrainingSet_InstRec_Balanced(const int trial, std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames);

		/** \brief Splitting for instance recognition (balanced) as described in:
		*  A. Petrelli, L. Di Stefano "Learning to Weight Color And Depth for RGB-D Image Search" ECCV, 2016
		*/
        size_t getTestSet_InstRec_Balanced(const int trial, std::vector< std::pair< std::string, std::vector<std::string> > > &vTestSet, const bool getAbsNames);


        size_t GetAllAbsFilenames(const int nMaxCategories, const int nMaxInstances, const int nMaxInstanceViews, std::vector< std::string > &vAbsFilenames);

		/** \brief Return all the dataset partitioned in categories.
		*
		* Used by HyperDataset class.
		*/
        static std::string GetImageFilename_AbsFull(const std::string &absFilename_withoutSuffix, const bool depth);

		/** \brief Return all the dataset partitioned in instances.
		*
		* Used by HyperDataset class.
		*/
        static void GetImageFilename_AbsFull(std::vector< std::pair< std::string, std::vector<std::string> > > &vDataSet, const bool depth);

         /** \brief Return a depth map (1-channel image of short data)
         * \param[in] convertToMillimeters false measures are in 100um
         */
        static cv::Mat ReadDepthMap(const std::string &absFilename);

         /** \brief Return a range map (3-channel image (x,y,z) of float data)
         * \param[in] convertToMillimeters false measures are in 100um
         */
        static cv::Mat ReadRangeMap(const std::string &absFilename);


        virtual cv::Mat ReadImage(const std::string &absFilename_withoutSuffix, const ImageType imageType);

        virtual size_t GetTrainingSet(std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames);
        virtual size_t GetTestSet(std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames);

        virtual size_t GetAllAbsFilenames(std::vector< std::string > &vAbsFilenames);

        virtual int GetNumTrials(){return 10;};

		virtual void SetTrial(const int trial);

        using IDataset::GetTrial;
    };
}

#endif