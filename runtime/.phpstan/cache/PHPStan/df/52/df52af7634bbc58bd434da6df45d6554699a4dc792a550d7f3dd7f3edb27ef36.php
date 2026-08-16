<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/CrossValidation/Reports/ErrorAnalysis.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\CrossValidation\Reports\ErrorAnalysis
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-bebf573b724096927efc016d67ff4209e2e7fa238124e015044ff1aca2e6060b',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\CrossValidation\\Reports\\ErrorAnalysis',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/CrossValidation/Reports/ErrorAnalysis.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\CrossValidation\\Reports',
    'name' => 'Rubix\\ML\\CrossValidation\\Reports\\ErrorAnalysis',
    'shortName' => 'ErrorAnalysis',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Error Analysis
 *
 * The Error Analysis report measures the differences between the predicted and target values
 * of a regression problem using multiple error measurements (MAE, MSE, RMSE, MAPE, etc.) as
 * well as statistics regarding the distribution of errors.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 26,
    'endLine' => 100,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Rubix\\ML\\CrossValidation\\Reports\\ReportGenerator',
    ),
    'traitClassNames' => 
    array (
    ),
    'immediateConstants' => 
    array (
    ),
    'immediateProperties' => 
    array (
    ),
    'immediateMethods' => 
    array (
      'compatibility' => 
      array (
        'name' => 'compatibility',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'array',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * The estimator types that this report is compatible with.
 *
 * @internal
 *
 * @return list<EstimatorType>
 */',
        'startLine' => 35,
        'endLine' => 40,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\CrossValidation\\Reports',
        'declaringClassName' => 'Rubix\\ML\\CrossValidation\\Reports\\ErrorAnalysis',
        'implementingClassName' => 'Rubix\\ML\\CrossValidation\\Reports\\ErrorAnalysis',
        'currentClassName' => 'Rubix\\ML\\CrossValidation\\Reports\\ErrorAnalysis',
        'aliasName' => NULL,
      ),
      'generate' => 
      array (
        'name' => 'generate',
        'parameters' => 
        array (
          'predictions' => 
          array (
            'name' => 'predictions',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'array',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 49,
            'endLine' => 49,
            'startColumn' => 30,
            'endColumn' => 47,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'labels' => 
          array (
            'name' => 'labels',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'array',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 49,
            'endLine' => 49,
            'startColumn' => 50,
            'endColumn' => 62,
            'parameterIndex' => 1,
            'isOptional' => false,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Rubix\\ML\\Report',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Generate the report.
 *
 * @param list<int|float> $predictions
 * @param list<int|float> $labels
 * @return Report
 */',
        'startLine' => 49,
        'endLine' => 99,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\CrossValidation\\Reports',
        'declaringClassName' => 'Rubix\\ML\\CrossValidation\\Reports\\ErrorAnalysis',
        'implementingClassName' => 'Rubix\\ML\\CrossValidation\\Reports\\ErrorAnalysis',
        'currentClassName' => 'Rubix\\ML\\CrossValidation\\Reports\\ErrorAnalysis',
        'aliasName' => NULL,
      ),
    ),
    'traitsData' => 
    array (
      'aliases' => 
      array (
      ),
      'modifiers' => 
      array (
      ),
      'precedences' => 
      array (
      ),
      'hashes' => 
      array (
      ),
    ),
  ),
));