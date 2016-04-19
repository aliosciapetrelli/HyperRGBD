#ifndef HYPERRGBD_WASHINGTON_H
#define HYPERRGBD_WASHINGTON_H

#include "hyperrgbdDefines.h"

#include "hyperrgbdIDataset.h"


namespace HyperRGBD
{

    /** \brief Wrapper of the dataset introduced in:
    *  K. Lai, L. Bo, X. Ren, and D. Fox. "A Large-Scale Hierarchical Multi-View RGB-D Object Dataset" ICRA 2011, May 2011.
    *
    * \author Alioscia Petrelli
    */
    class HYPER_RGBD_API Washington : virtual public IDataset 
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


		/** \brief Absolute filename for testinstance_ids.txt file (used for category recognition splitting).*/
        std::string m_absLeaveOneOutCategoryInstances;

		/** \brief Absolute path for dataset of all cropped images (not used).*/
        std::string m_absCropDir;

		/** \brief Absolute path for evaluation dataset (used in papers).*/
        std::string m_absEvalDir;

		/** \brief Data in testinstance_ids.txt file.
		*
		* Each element denotes a trial and the set of instances used as test set
		*/
        std::vector<std::set<std::string> > m_vsLeaveOneOutCategoryInstances;

		/** \brief Data used for "Alternating Contiguous Frames" splitting for instance recognition.
		*
		* Each element of the map denotes an instance. Precisely: [ (video idx, firstView idx, last(+1)View idx) of first left out sequence, (video idx, firstView idx, last(+1)View idx) of second left out sequence ]  video idx = 1,2,4 
		*/
        std::map< std::string, std::vector<int> > m_mvLeaveTwoOutInstanceSequences;	//instance recognition Alternating Contiguous Frames: [instances]

		/** \brief Training set as vector of pairs <label, vector of filenames>.*/
        std::vector< std::pair< std::string, std::vector<std::string> > > m_vTrainingSet;
        
		/** \brief Test set as vector of pairs <label, vector of filenames>.*/
		std::vector< std::pair< std::string, std::vector<std::string> > > m_vTestSet;

		/** \brief Empty constructor */
        Washington();

		/** \brief Constructor that sets dataset root*/
        Washington(const std::string &absDatasetRoot);

		/** \brief Initalization*/
        virtual void Init();

		/** \brief Set m_absLeaveOneOutCategoryInstances, m_absCropDir and m_absEvalDir
		*
		* Assume that:
		*     evalution dataset is in absDatasetRoot/rgbd-dataset_eval
		*     cropped dataset is in absDatasetRoot/rgbd-dataset
		*     testinstance_ids.txt file is in absDatasetRoot
		*/
        virtual void SetRoot(const std::string &absDatasetRoot);

		/** \brief parse absLeaveOneOutCategoryInstances and populate vsLeaveOneOutCategoryInstances */
        void ParseLeaveOneOutCategoryInstances(const std::string &absLeaveOneOutCategoryInstances, std::vector<std::set<std::string> > &vsLeaveOneOutCategoryInstances);

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
		*   K. Lai, L. Bo, X. Ren, and D. Fox. "A Large-Scale Hierarchical Multi-View RGB-D Object Dataset" ICRA 2011, May 2011.
		*/
        size_t GetTrainingSet_Category(const int trial, std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames);

		/** \brief Splitting for category recognition as described in:
		*   K. Lai, L. Bo, X. Ren, and D. Fox. "A Large-Scale Hierarchical Multi-View RGB-D Object Dataset" ICRA 2011, May 2011.
		*/
        size_t GetTestSet_Category(const int trial, std::vector< std::pair< std::string, std::vector<std::string> > > &vTestSet, const bool getAbsNames);

		/** \brief Splitting for instance recognition (Alternating Contiguous Frames) as described in:
		*   K. Lai, L. Bo, X. Ren, and D. Fox. "A Large-Scale Hierarchical Multi-View RGB-D Object Dataset" ICRA 2011, May 2011.
		*/
        size_t GetTraining_And_Test_Set_InstanceRec_AlternatingContiguousFrames(const int trial, std::map< std::string, std::vector<int> > &mvLeaveTwoOutInstanceSequences, std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, std::vector< std::pair< std::string, std::vector<std::string> > > &vTestSet, const bool getAbsNames);

		/** \brief Splitting for instance recognition (Leave Sequence Out) as described in:
		*   K. Lai, L. Bo, X. Ren, and D. Fox. "A Large-Scale Hierarchical Multi-View RGB-D Object Dataset" ICRA 2011, May 2011.
		*/
        size_t GetTrainingSet_InstanceRec_LeaveSequenceOut(std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames);

		/** \brief Splitting for instance recognition (Leave Sequence Out) as described in:
		*   K. Lai, L. Bo, X. Ren, and D. Fox. "A Large-Scale Hierarchical Multi-View RGB-D Object Dataset" ICRA 2011, May 2011.
		*/
        size_t GetTestSet_InstanceRec_LeaveSequenceOut(std::vector< std::pair< std::string, std::vector<std::string> > > &vTestSet, const bool getAbsNames);

        size_t GetTrainingSet(const std::string &evalType, std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames);
        size_t GetTestSet(const std::string &evalType, std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames);


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

		/** \brief Return the absolute path of the file containing the top left coordinates used for cropping the image absFilename_withoutSuffix (used for generating the depth map)*/
        static std::string GetTopLeftLocFilename_AbsFull(const std::string &absFilename_withoutSuffix){return absFilename_withoutSuffix + "_loc.txt";};

		/** \brief Return the top left coordinates in filename (used for generating the depth map)*/
        static void ReadTopLeftLoc(const std::string &filename, std::vector<float> &topLeft );

		/** \brief Return a range map (3-channel image (x,y,z) of float data)
		* \param[in] convertToMillimeters false measures are in 100um
		*/
        cv::Mat ReadRangeMap(const std::string &absFilename_withoutSuffix);

        /** \brief Convert from depthmap (1-channel short image) to range image (3-channels x,y,z float image)
		* \param[in] topLeftDispl  top left coordinates loaded with ReadTopLeftLoc method
		*/
        cv::Mat DepthMap2RangeMap(const cv::Mat &depthMap, const std::vector<float> &topLeftDispl, double focal=570.3, double ycenter=240.0, double xcenter=320.0);
		

        virtual cv::Mat ReadImage(const std::string &absFilename_withoutSuffix, const ImageType imageType);

        virtual size_t GetTrainingSet(std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames);
        virtual size_t GetTestSet(std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames);

        virtual size_t GetAllAbsFilenames(std::vector< std::string > &vAbsFilenames);

        virtual std::string GetRoot(){return m_absEvalDir;}

        virtual int GetNumTrials(){return ((m_evalType == "InstRec_LSO")||(m_evalType == "InstRec_Full")||(m_evalType == "CatRec_Full"))?1:10;};

        using IDataset::SetTrial;
        using IDataset::GetTrial;
    };
}

#endif
