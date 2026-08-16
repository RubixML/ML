<?php declare(strict_types = 1);

// phpinternal-PHPStan\BetterReflection\Reflection\ReflectionClass-svm
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-dev-master@709e512-8.4',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\InternalLocatedSource',
      'data' => 
      array (
        'name' => 'svm',
        'filename' => 'phpstorm-stubs:svm/SVM.stub',
        'extensionName' => 'svm',
        'aliasName' => NULL,
      ),
    ),
    'namespace' => NULL,
    'name' => 'SVM',
    'shortName' => 'SVM',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Support Vector Machine Library
 * LibSVM is an efficient solver for SVM classification and regression problems. The svm extension wraps this in a PHP interface for easy use in PHP scripts.
 * @since 7.0
 * @link https://www.php.net/manual/en/class.svm.php
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 10,
    'endLine' => 156,
    'startColumn' => 5,
    'endColumn' => 5,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
    ),
    'traitClassNames' => 
    array (
    ),
    'immediateConstants' => 
    array (
      'C_SVC' => 
      array (
        'declaringClassName' => 'SVM',
        'implementingClassName' => 'SVM',
        'name' => 'C_SVC',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '0',
          'attributes' => 
          array (
            'startLine' => 16,
            'endLine' => 16,
            'startTokenPos' => 26,
            'startFilePos' => 483,
            'endTokenPos' => 26,
            'endFilePos' => 483,
          ),
        ),
        'docComment' => '/**
 * The basic C_SVC SVM type. The default, and a good starting point
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 16,
        'endLine' => 16,
        'startColumn' => 9,
        'endColumn' => 31,
      ),
      'NU_SVC' => 
      array (
        'declaringClassName' => 'SVM',
        'implementingClassName' => 'SVM',
        'name' => 'NU_SVC',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '1',
          'attributes' => 
          array (
            'startLine' => 20,
            'endLine' => 20,
            'startTokenPos' => 39,
            'startFilePos' => 612,
            'endTokenPos' => 39,
            'endFilePos' => 612,
          ),
        ),
        'docComment' => '/**
 * NU_SVC type uses a different, more flexible, error weighting
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 20,
        'endLine' => 20,
        'startColumn' => 9,
        'endColumn' => 32,
      ),
      'ONE_CLASS' => 
      array (
        'declaringClassName' => 'SVM',
        'implementingClassName' => 'SVM',
        'name' => 'ONE_CLASS',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '2',
          'attributes' => 
          array (
            'startLine' => 24,
            'endLine' => 24,
            'startTokenPos' => 52,
            'startFilePos' => 769,
            'endTokenPos' => 52,
            'endFilePos' => 769,
          ),
        ),
        'docComment' => '/**
 * One class SVM type. Train just on a single class, using outliers as negative examples
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 24,
        'endLine' => 24,
        'startColumn' => 9,
        'endColumn' => 35,
      ),
      'EPSILON_SVR' => 
      array (
        'declaringClassName' => 'SVM',
        'implementingClassName' => 'SVM',
        'name' => 'EPSILON_SVR',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '3',
          'attributes' => 
          array (
            'startLine' => 28,
            'endLine' => 28,
            'startTokenPos' => 65,
            'startFilePos' => 914,
            'endTokenPos' => 65,
            'endFilePos' => 914,
          ),
        ),
        'docComment' => '/**
 * A SVM type for regression (predicting a value rather than just a class)
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 28,
        'endLine' => 28,
        'startColumn' => 9,
        'endColumn' => 37,
      ),
      'NU_SVR' => 
      array (
        'declaringClassName' => 'SVM',
        'implementingClassName' => 'SVM',
        'name' => 'NU_SVR',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '4',
          'attributes' => 
          array (
            'startLine' => 32,
            'endLine' => 32,
            'startTokenPos' => 78,
            'startFilePos' => 1013,
            'endTokenPos' => 78,
            'endFilePos' => 1013,
          ),
        ),
        'docComment' => '/**
 * A NU style SVM regression type
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 32,
        'endLine' => 32,
        'startColumn' => 9,
        'endColumn' => 32,
      ),
      'KERNEL_LINEAR' => 
      array (
        'declaringClassName' => 'SVM',
        'implementingClassName' => 'SVM',
        'name' => 'KERNEL_LINEAR',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '0',
          'attributes' => 
          array (
            'startLine' => 36,
            'endLine' => 36,
            'startTokenPos' => 91,
            'startFilePos' => 1166,
            'endTokenPos' => 91,
            'endFilePos' => 1166,
          ),
        ),
        'docComment' => '/**
 * A very simple kernel, can work well on large document classification problems
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 36,
        'endLine' => 36,
        'startColumn' => 9,
        'endColumn' => 39,
      ),
      'KERNEL_POLY' => 
      array (
        'declaringClassName' => 'SVM',
        'implementingClassName' => 'SVM',
        'name' => 'KERNEL_POLY',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '1',
          'attributes' => 
          array (
            'startLine' => 40,
            'endLine' => 40,
            'startTokenPos' => 104,
            'startFilePos' => 1259,
            'endTokenPos' => 104,
            'endFilePos' => 1259,
          ),
        ),
        'docComment' => '/**
 * A polynomial kernel
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 40,
        'endLine' => 40,
        'startColumn' => 9,
        'endColumn' => 37,
      ),
      'KERNEL_RBF' => 
      array (
        'declaringClassName' => 'SVM',
        'implementingClassName' => 'SVM',
        'name' => 'KERNEL_RBF',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '2',
          'attributes' => 
          array (
            'startLine' => 44,
            'endLine' => 44,
            'startTokenPos' => 117,
            'startFilePos' => 1437,
            'endTokenPos' => 117,
            'endFilePos' => 1437,
          ),
        ),
        'docComment' => '/**
 * The common Gaussian RBD kernel. Handles non-linear problems well and is a good default for classification
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 44,
        'endLine' => 44,
        'startColumn' => 9,
        'endColumn' => 36,
      ),
      'KERNEL_SIGMOID' => 
      array (
        'declaringClassName' => 'SVM',
        'implementingClassName' => 'SVM',
        'name' => 'KERNEL_SIGMOID',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '3',
          'attributes' => 
          array (
            'startLine' => 48,
            'endLine' => 48,
            'startTokenPos' => 130,
            'startFilePos' => 1635,
            'endTokenPos' => 130,
            'endFilePos' => 1635,
          ),
        ),
        'docComment' => '/**
 * A kernel based on the sigmoid function. Using this makes the SVM very similar to a two layer sigmoid based neural network
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 48,
        'endLine' => 48,
        'startColumn' => 9,
        'endColumn' => 40,
      ),
      'KERNEL_PRECOMPUTED' => 
      array (
        'declaringClassName' => 'SVM',
        'implementingClassName' => 'SVM',
        'name' => 'KERNEL_PRECOMPUTED',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '4',
          'attributes' => 
          array (
            'startLine' => 52,
            'endLine' => 52,
            'startTokenPos' => 143,
            'startFilePos' => 1761,
            'endTokenPos' => 143,
            'endFilePos' => 1761,
          ),
        ),
        'docComment' => '/**
 * A precomputed kernel - currently unsupported.
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 52,
        'endLine' => 52,
        'startColumn' => 9,
        'endColumn' => 44,
      ),
      'OPT_TYPE' => 
      array (
        'declaringClassName' => 'SVM',
        'implementingClassName' => 'SVM',
        'name' => 'OPT_TYPE',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '101',
          'attributes' => 
          array (
            'startLine' => 56,
            'endLine' => 56,
            'startTokenPos' => 156,
            'startFilePos' => 1864,
            'endTokenPos' => 156,
            'endFilePos' => 1866,
          ),
        ),
        'docComment' => '/**
 * The options key for the SVM type
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 56,
        'endLine' => 56,
        'startColumn' => 9,
        'endColumn' => 36,
      ),
      'OPT_KERNEL_TYPE' => 
      array (
        'declaringClassName' => 'SVM',
        'implementingClassName' => 'SVM',
        'name' => 'OPT_KERNEL_TYPE',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '102',
          'attributes' => 
          array (
            'startLine' => 60,
            'endLine' => 60,
            'startTokenPos' => 169,
            'startFilePos' => 1979,
            'endTokenPos' => 169,
            'endFilePos' => 1981,
          ),
        ),
        'docComment' => '/**
 * The options key for the kernel type
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 60,
        'endLine' => 60,
        'startColumn' => 9,
        'endColumn' => 43,
      ),
      'OPT_DEGREE' => 
      array (
        'declaringClassName' => 'SVM',
        'implementingClassName' => 'SVM',
        'name' => 'OPT_DEGREE',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '103',
          'attributes' => 
          array (
            'startLine' => 61,
            'endLine' => 61,
            'startTokenPos' => 180,
            'startFilePos' => 2018,
            'endTokenPos' => 180,
            'endFilePos' => 2020,
          ),
        ),
        'docComment' => NULL,
        'attributes' => 
        array (
        ),
        'startLine' => 61,
        'endLine' => 61,
        'startColumn' => 9,
        'endColumn' => 38,
      ),
      'OPT_SHRINKING' => 
      array (
        'declaringClassName' => 'SVM',
        'implementingClassName' => 'SVM',
        'name' => 'OPT_SHRINKING',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '301',
          'attributes' => 
          array (
            'startLine' => 65,
            'endLine' => 65,
            'startTokenPos' => 193,
            'startFilePos' => 2168,
            'endTokenPos' => 193,
            'endFilePos' => 2170,
          ),
        ),
        'docComment' => '/**
 * Training parameter, boolean, for whether to use the shrinking heuristics
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 65,
        'endLine' => 65,
        'startColumn' => 9,
        'endColumn' => 41,
      ),
      'OPT_PROPABILITY' => 
      array (
        'declaringClassName' => 'SVM',
        'implementingClassName' => 'SVM',
        'name' => 'OPT_PROPABILITY',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '105',
          'attributes' => 
          array (
            'startLine' => 69,
            'endLine' => 69,
            'startTokenPos' => 206,
            'startFilePos' => 2329,
            'endTokenPos' => 206,
            'endFilePos' => 2331,
          ),
        ),
        'docComment' => '/**
 * Training parameter, boolean, for whether to collect and use probability estimates
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 69,
        'endLine' => 69,
        'startColumn' => 9,
        'endColumn' => 43,
      ),
      'OPT_GAMMA' => 
      array (
        'declaringClassName' => 'SVM',
        'implementingClassName' => 'SVM',
        'name' => 'OPT_GAMMA',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '201',
          'attributes' => 
          array (
            'startLine' => 73,
            'endLine' => 73,
            'startTokenPos' => 219,
            'startFilePos' => 2462,
            'endTokenPos' => 219,
            'endFilePos' => 2464,
          ),
        ),
        'docComment' => '/**
 * Algorithm parameter for Poly, RBF and Sigmoid kernel types.
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 73,
        'endLine' => 73,
        'startColumn' => 9,
        'endColumn' => 37,
      ),
      'OPT_NU' => 
      array (
        'declaringClassName' => 'SVM',
        'implementingClassName' => 'SVM',
        'name' => 'OPT_NU',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '202',
          'attributes' => 
          array (
            'startLine' => 77,
            'endLine' => 77,
            'startTokenPos' => 232,
            'startFilePos' => 2600,
            'endTokenPos' => 232,
            'endFilePos' => 2602,
          ),
        ),
        'docComment' => '/**
 * The option key for the nu parameter, only used in the NU_ SVM types
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 77,
        'endLine' => 77,
        'startColumn' => 9,
        'endColumn' => 34,
      ),
      'OPT_EPS' => 
      array (
        'declaringClassName' => 'SVM',
        'implementingClassName' => 'SVM',
        'name' => 'OPT_EPS',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '203',
          'attributes' => 
          array (
            'startLine' => 81,
            'endLine' => 81,
            'startTokenPos' => 245,
            'startFilePos' => 2740,
            'endTokenPos' => 245,
            'endFilePos' => 2742,
          ),
        ),
        'docComment' => '/**
 * The option key for the Epsilon parameter, used in epsilon regression
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 81,
        'endLine' => 81,
        'startColumn' => 9,
        'endColumn' => 35,
      ),
      'OPT_P' => 
      array (
        'declaringClassName' => 'SVM',
        'implementingClassName' => 'SVM',
        'name' => 'OPT_P',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '204',
          'attributes' => 
          array (
            'startLine' => 85,
            'endLine' => 85,
            'startTokenPos' => 258,
            'startFilePos' => 2860,
            'endTokenPos' => 258,
            'endFilePos' => 2862,
          ),
        ),
        'docComment' => '/**
 * Training parameter used by Episilon SVR regression
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 85,
        'endLine' => 85,
        'startColumn' => 9,
        'endColumn' => 33,
      ),
      'OPT_COEF_ZERO' => 
      array (
        'declaringClassName' => 'SVM',
        'implementingClassName' => 'SVM',
        'name' => 'OPT_COEF_ZERO',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '205',
          'attributes' => 
          array (
            'startLine' => 89,
            'endLine' => 89,
            'startTokenPos' => 271,
            'startFilePos' => 2986,
            'endTokenPos' => 271,
            'endFilePos' => 2988,
          ),
        ),
        'docComment' => '/**
 * Algorithm parameter for poly and sigmoid kernels
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 89,
        'endLine' => 89,
        'startColumn' => 9,
        'endColumn' => 41,
      ),
      'OPT_C' => 
      array (
        'declaringClassName' => 'SVM',
        'implementingClassName' => 'SVM',
        'name' => 'OPT_C',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '206',
          'attributes' => 
          array (
            'startLine' => 93,
            'endLine' => 93,
            'startTokenPos' => 284,
            'startFilePos' => 3206,
            'endTokenPos' => 284,
            'endFilePos' => 3208,
          ),
        ),
        'docComment' => '/**
 * The option for the cost parameter that controls tradeoff between errors and generality - effectively the penalty for misclassifying training examples.
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 93,
        'endLine' => 93,
        'startColumn' => 9,
        'endColumn' => 33,
      ),
      'OPT_CACHE_SIZE' => 
      array (
        'declaringClassName' => 'SVM',
        'implementingClassName' => 'SVM',
        'name' => 'OPT_CACHE_SIZE',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '207',
          'attributes' => 
          array (
            'startLine' => 97,
            'endLine' => 97,
            'startTokenPos' => 297,
            'startFilePos' => 3309,
            'endTokenPos' => 297,
            'endFilePos' => 3311,
          ),
        ),
        'docComment' => '/**
 * Memory cache size, in MB
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 97,
        'endLine' => 97,
        'startColumn' => 9,
        'endColumn' => 42,
      ),
    ),
    'immediateProperties' => 
    array (
    ),
    'immediateMethods' => 
    array (
      '__construct' => 
      array (
        'name' => '__construct',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => NULL,
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Construct a new SVM object
 *
 * Constructs a new SVM object ready to accept training data.
 * @throws SVMException Throws SVMException if the libsvm library could not be loaded
 * @link https://www.php.net/manual/en/svm.construct.php
 */',
        'startLine' => 106,
        'endLine' => 108,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'SVM',
        'implementingClassName' => 'SVM',
        'currentClassName' => 'SVM',
        'aliasName' => NULL,
      ),
      'crossvalidate' => 
      array (
        'name' => 'crossvalidate',
        'parameters' => 
        array (
          'problem' => 
          array (
            'name' => 'problem',
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
            'startLine' => 118,
            'endLine' => 118,
            'startColumn' => 39,
            'endColumn' => 52,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'number_of_folds' => 
          array (
            'name' => 'number_of_folds',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'int',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 118,
            'endLine' => 118,
            'startColumn' => 55,
            'endColumn' => 74,
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
            'name' => 'float',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Test training params on subsets of the training data
 *
 * Crossvalidate can be used to test the effectiveness of the current parameter set on a subset of the training data. Given a problem set and a n "folds", it separates the problem set into n subsets, and the repeatedly trains on one subset and tests on another. While the accuracy will generally be lower than a SVM trained on the enter data set, the accuracy score returned should be relatively useful, so it can be used to test different training parameters.
 * @param array $problem The problem data. This can either be in the form of an array, the URL of an SVMLight formatted file, or a stream to an opened SVMLight formatted datasource.
 * @param int $number_of_folds The number of sets the data should be divided into and cross tested. A higher number means smaller training sets and less reliability. 5 is a good number to start with.
 * @return float The correct percentage, expressed as a floating point number from 0-1. In the case of NU_SVC or EPSILON_SVR kernels the mean squared error will returned instead.
 * @link https://www.php.net/manual/en/svm.crossvalidate.php
 */',
        'startLine' => 118,
        'endLine' => 120,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'SVM',
        'implementingClassName' => 'SVM',
        'currentClassName' => 'SVM',
        'aliasName' => NULL,
      ),
      'getOptions' => 
      array (
        'name' => 'getOptions',
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
 * Return the current training parameters
 *
 * Retrieve an array containing the training parameters. The parameters will be keyed on the predefined SVM constants.
 * @return array Returns an array of configuration settings.
 * @link https://www.php.net/manual/en/svm.getoptions.php
 */',
        'startLine' => 128,
        'endLine' => 130,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'SVM',
        'implementingClassName' => 'SVM',
        'currentClassName' => 'SVM',
        'aliasName' => NULL,
      ),
      'setOptions' => 
      array (
        'name' => 'setOptions',
        'parameters' => 
        array (
          'params' => 
          array (
            'name' => 'params',
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
            'startLine' => 140,
            'endLine' => 140,
            'startColumn' => 36,
            'endColumn' => 48,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'bool',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Set training parameters
 *
 * Set one or more training parameters.
 * @param array $params An array of training parameters, keyed on the SVM constants.
 * @return bool Return true on success, throws SVMException on error.
 * @throws SVMException
 * @link https://www.php.net/manual/en/svm.setoptions.php
 */',
        'startLine' => 140,
        'endLine' => 142,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'SVM',
        'implementingClassName' => 'SVM',
        'currentClassName' => 'SVM',
        'aliasName' => NULL,
      ),
      'train' => 
      array (
        'name' => 'train',
        'parameters' => 
        array (
          'problem' => 
          array (
            'name' => 'problem',
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
            'startLine' => 153,
            'endLine' => 153,
            'startColumn' => 31,
            'endColumn' => 44,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'weights' => 
          array (
            'name' => 'weights',
            'default' => 
            array (
              'code' => '\\null',
              'attributes' => 
              array (
                'startLine' => 153,
                'endLine' => 153,
                'startTokenPos' => 398,
                'startFilePos' => 7192,
                'endTokenPos' => 398,
                'endFilePos' => 7195,
              ),
            ),
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionUnionType',
              'data' => 
              array (
                'types' => 
                array (
                  0 => 
                  array (
                    'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
                    'data' => 
                    array (
                      'name' => 'array',
                      'isIdentifier' => true,
                    ),
                  ),
                  1 => 
                  array (
                    'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
                    'data' => 
                    array (
                      'name' => 'null',
                      'isIdentifier' => true,
                    ),
                  ),
                ),
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 153,
            'endLine' => 153,
            'startColumn' => 47,
            'endColumn' => 68,
            'parameterIndex' => 1,
            'isOptional' => true,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'SVMModel',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Create a SVMModel based on training data
 *
 * Train a support vector machine based on the supplied training data.
 * @param array $problem The problem can be provided in three different ways. An array, where the data should start with the class label (usually 1 or -1) then followed by a sparse data set of dimension => data pairs. A URL to a file containing a SVM Light formatted problem, with the each line being a new training example, the start of each line containing the class (1, -1) then a series of tab separated data values shows as key:value. A opened stream pointing to a data source formatted as in the file above.
 * @param array|null $weights Weights are an optional set of weighting parameters for the different classes, to help account for unbalanced training sets. For example, if the classes were 1 and -1, and -1 had significantly more example than one, the weight for -1 could be 0.5. Weights should be in the range 0-1.
 * @return SVMModel Returns an SVMModel that can be used to classify previously unseen data. Throws SVMException on error
 * @throws SMVException
 * @link https://www.php.net/manual/en/svm.train.php
 */',
        'startLine' => 153,
        'endLine' => 155,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'SVM',
        'implementingClassName' => 'SVM',
        'currentClassName' => 'SVM',
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