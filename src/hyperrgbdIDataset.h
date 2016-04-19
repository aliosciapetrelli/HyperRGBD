#ifndef HYPERRGBD_IDATASET_H
#define HYPERRGBD_IDATASET_H

#include "hyperrgbdDefines.h"

#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <set>

#include "opencv2/opencv.hpp"



/** \brief Framework for the aggregation of RGBD datasets, as described in:
*
*  - A. Petrelli, L. Di Stefano,
*    "Learning to Weight Color And Depth for RGB-D Image Search",
*    ECCV, 2016
*
* \author Alioscia Petrelli
*/
namespace HyperRGBD
{

    enum ImageType {IMAGETYPE_AS_IS, IMAGETYPE_RGB, IMAGETYPE_GREY, IMAGETYPE_DEPTHMAP, IMAGETYPE_RANGEMAP, IMAGETYPE_MASK, IMAGETYPE_POSE};


    /** \brief Interface every dataset has to implement
    * \author Alioscia Petrelli
    */
    class HYPER_RGBD_API IDataset
    {

    protected:

        /** \brief absolute path of the dataset. */
        std::string m_absDatasetRoot;

        /** \brief index of the processed trial. It defines the splitting in training and test set*/
        int m_trial;

        /** \brief Store, for each instance in a category, the starting index of that instance in the vector of views*/
        std::vector< std::pair< std::string, std::vector< size_t > > > m_vDataSet_InstBegins;

    public:
        /** \brief Splitting stragegy. It could be Category recognition, Instance recogntion or others*/
        std::string m_evalType;


        virtual void Init() = 0;

        virtual ~IDataset(){};

        /** \brief Return a specific image of the dataset.
        *
        * \param[in] absFilename_template absolute path of an image of the dataset. It is usually returned by GetTrainingSet, GetTestSet or GetAllAbsFilenames
        * \param[in] imageType format of returned image (e.g. RGB, depth map, 3D point cloud, mask image etc.). 
        */
        virtual cv::Mat ReadImage(const std::string &absFilename_template, const ImageType imageType) = 0;

        /** \brief Return the training set as a vector of pairs: (label, vector of filenames)
        *
        * \param[out] vTrainingSet training set as a vector of pairs: (label, vector of filenames).  
        * \param[in] getAbsNames if true, filenames in vTrainingSet are returned as absolute paths, otherwise they do not include the m_absDatasetRoot string.  
        */
        virtual size_t GetTrainingSet(std::vector< std::pair< std::string, std::vector<std::string> > > &vTrainingSet, const bool getAbsNames) = 0;

        /** \brief Return the test set as a vector of pairs: (label, vector of filenames)
        *
        * \param[out] vTestSet test set as a vector of pairs: (label, vector of filenames).  
        * \param[in] getAbsNames if true, filenames in vTestSet are returned as absolute paths, otherwise they do not include the m_absDatasetRoot string.  
        */
        virtual size_t GetTestSet(std::vector< std::pair< std::string, std::vector<std::string> > > &vTestSet, const bool getAbsNames) = 0;

        /** \brief Return all the images of the dataset as a vector of absolute filenames
        *
        * \param[out] vAbsFilenames absolute filenames of all the images of the dataset.  
        */        
        virtual size_t GetAllAbsFilenames(std::vector< std::string > &vAbsFilenames) = 0;

        /** \brief Return m_absDatasetRoot
        */  
        virtual std::string GetRoot(){return m_absDatasetRoot;};

        /** \brief Set m_absDatasetRoot
        * \param[in] absDatasetRoot absolute path of the dataset.
        */  
        virtual void SetRoot(const std::string &absDatasetRoot)
        {
            m_absDatasetRoot = absDatasetRoot;
            replace(m_absDatasetRoot.begin(), m_absDatasetRoot.end(), '\\' , '/');
        };


        /** \brief Return the evaluation type.
        */  
        virtual std::string GetEvalType(){return m_evalType;}

        /** \brief Set the evaluation type.
        * \param[in] trial evaluation type.
        */
        virtual void SetEvalType(const std::string evalType){m_evalType = evalType;}


        /** \brief Return the current index of trial.
        */  
        virtual int GetTrial(){return m_trial;}

        /** \brief Set the index of the trial to process.
        * \param[in] trial index of the trial to process.
        */
        virtual void SetTrial(const int trial){m_trial = trial;}

        /** \brief Return the number of trials to process. As an example, 10 in the case of Category recognition for the Washington dataset.
        */ 
        virtual int GetNumTrials() = 0;

        /** \brief After the GetTrainingSet method is called in the case of category recognition, return a vector of pairs <category, vector of starting indices> denoting, for each instance in a category, the starting index of that instance in the vector of views*/
        virtual std::vector< std::pair< std::string, std::vector< size_t > > > &GetInstBegins(){ return m_vDataSet_InstBegins;};

    };






}

#endif