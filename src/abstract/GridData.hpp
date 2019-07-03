#ifndef GRIDDATA_HPP
#define GRIDDATA_HPP
#include <string>
#include <iostream>

//   different data layouts to choose from  
#define NONE
//#define FULL
//#define ONELAYER
//#define TWOLAYER
//#define TWOTWOLAYER
//#define STRUCTARR

// use local memory? 
#define LOCAL

// other definitions could go here ! 


/*
 *  Parent template class, outlining what functions MUST be defined
 *  Or they won't be included in the header file built for the kernel 
 */

class GridData 
{

    public:
       
        GridData() {}
        
        ~GridData() {}

        virtual std::string getInitialDeclaration() = 0;
        
        virtual std::string getNeighbourSumFunction() = 0;

        virtual std::string addToPointFunction() = 0;

        virtual std::string addToPointWithIndexFunction() = 0;

        virtual std::string setPointFunction() = 0;

        virtual std::string getSourceIndexFunction() = 0;

        virtual std::string getReceiverIndexFunction() = 0;

        virtual std::string setReceiverPointFunction() = 0;

        virtual std::string setPointWithIndexFunction() = 0;
        
        virtual std::string getValueAtPointFunction() = 0;
       
        virtual std::string getValueAtPointWithIndexFunction() = 0;
        
        virtual std::string CalculateIndexFunction() = 0;
        
        virtual std::string getNumberOfNeighboursFunction() = 0;
        
        virtual std::string isIndexOnBoundaryFunction() = 0;
        
        virtual std::string calculateBoundaryConditionChangeFunction() = 0;

        virtual std::string UpdateBoundaryConditionsFunction() = 0;

        virtual std::string InitialiseBoundaryConditionsFunction() = 0;

        virtual std::string calculateBoundaryCondition1Function() = 0;
        
        virtual std::string calculateBoundaryCondition2Function() = 0;
        
        virtual std::string UpdateStencilFunction() = 0;
        
        virtual std::string ApplyStencilUpdateFunction() = 0;
        
        virtual std::string isOnGridFunction() = 0;
        
        virtual std::string getXDimFunction() = 0;
        
        virtual std::string getYDimFunction() = 0;
        
        virtual std::string getZDimFunction() = 0;
        
        virtual std::string getAreaFunction() = 0;
        
        virtual std::string getKernelXIdxFunction() = 0;
        
        virtual std::string getKernelYIdxFunction() = 0;
        
        virtual std::string getKernelZIdxFunction() = 0;

        virtual std::string SetupFunction() = 0;

        virtual std::string BreakdownFunction() = 0;

        virtual std::string OptimiseFunction() = 0;
        
};

#endif
