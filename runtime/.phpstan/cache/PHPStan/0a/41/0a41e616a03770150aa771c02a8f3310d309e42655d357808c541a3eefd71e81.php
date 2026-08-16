<?php declare(strict_types = 1);

// phpinternal-PHPStan\BetterReflection\Reflection\ReflectionClass-svmmodel
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-dev-master@709e512-8.4',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\InternalLocatedSource',
      'data' => 
      array (
        'name' => 'svmmodel',
        'filename' => 'phpstorm-stubs:svm/SVMModel.stub',
        'extensionName' => 'svm',
        'aliasName' => NULL,
      ),
    ),
    'namespace' => NULL,
    'name' => 'SVMModel',
    'shortName' => 'SVMModel',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * The SVMModel is the end result of the training process. It can be used to classify previously unseen data.
 * @since 7.0
 * @link https://www.php.net/manual/en/class.svmmodel.php
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 9,
    'endLine' => 116,
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
    ),
    'immediateProperties' => 
    array (
    ),
    'immediateMethods' => 
    array (
      'checkProbabilityModel' => 
      array (
        'name' => 'checkProbabilityModel',
        'parameters' => 
        array (
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
 * Returns true if the model has probability information
 *
 * @return bool Return a boolean value
 * @link https://www.php.net/manual/en/svmmodel.checkprobabilitymodel.php
 */',
        'startLine' => 18,
        'endLine' => 20,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'SVMModel',
        'implementingClassName' => 'SVMModel',
        'currentClassName' => 'SVMModel',
        'aliasName' => NULL,
      ),
      '__construct' => 
      array (
        'name' => '__construct',
        'parameters' => 
        array (
          'filename' => 
          array (
            'name' => 'filename',
            'default' => 
            array (
              'code' => '\'\'',
              'attributes' => 
              array (
                'startLine' => 29,
                'endLine' => 29,
                'startTokenPos' => 47,
                'startFilePos' => 1062,
                'endTokenPos' => 47,
                'endFilePos' => 1063,
              ),
            ),
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'string',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 29,
            'endLine' => 29,
            'startColumn' => 37,
            'endColumn' => 57,
            'parameterIndex' => 0,
            'isOptional' => true,
          ),
        ),
        'returnsReference' => false,
        'returnType' => NULL,
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Construct a new SVMModel
 *
 * Build a new SVMModel. Models will usually be created from the SVM::train function, but then saved models may be restored directly.
 * @param string $filename The filename for the saved model file this model should load.
 * @throws Throws SVMException on error
 * @link https://www.php.net/manual/en/svmmodel.construct.php
 */',
        'startLine' => 29,
        'endLine' => 31,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'SVMModel',
        'implementingClassName' => 'SVMModel',
        'currentClassName' => 'SVMModel',
        'aliasName' => NULL,
      ),
      'getLabels' => 
      array (
        'name' => 'getLabels',
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
 * Get the labels the model was trained on
 *
 * Return an array of labels that the model was trained on. For regression and one class models an empty array is returned.
 * @return array Return an array of labels
 * @link https://www.php.net/manual/en/svmmodel.getlabels.php
 */',
        'startLine' => 39,
        'endLine' => 41,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'SVMModel',
        'implementingClassName' => 'SVMModel',
        'currentClassName' => 'SVMModel',
        'aliasName' => NULL,
      ),
      'getNrClass' => 
      array (
        'name' => 'getNrClass',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'int',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Returns the number of classes the model was trained with
 *
 * Returns the number of classes the model was trained with, will return 2 for one class and regression models.
 * @return int Return an integer number of classes
 * @link https://www.php.net/manual/en/svmmodel.getnrclass.php
 */',
        'startLine' => 49,
        'endLine' => 51,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'SVMModel',
        'implementingClassName' => 'SVMModel',
        'currentClassName' => 'SVMModel',
        'aliasName' => NULL,
      ),
      'getSvmType' => 
      array (
        'name' => 'getSvmType',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'int',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Get the SVM type the model was trained with
 *
 * Returns an integer value representing the type of the SVM model used, e.g SVM::C_SVC.
 * @return int Return an integer SVM type
 * @link https://www.php.net/manual/en/svmmodel.getsvmtype.php
 */',
        'startLine' => 59,
        'endLine' => 61,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'SVMModel',
        'implementingClassName' => 'SVMModel',
        'currentClassName' => 'SVMModel',
        'aliasName' => NULL,
      ),
      'getSvrProbability' => 
      array (
        'name' => 'getSvrProbability',
        'parameters' => 
        array (
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
 * Get the sigma value for regression types
 *
 * For regression models, returns a sigma value. If there is no probability information or the model is not SVR, 0 is returned.
 * @return float Returns a sigma value
 * @link https://www.php.net/manual/en/svmmodel.getsvrprobability.php
 */',
        'startLine' => 69,
        'endLine' => 71,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'SVMModel',
        'implementingClassName' => 'SVMModel',
        'currentClassName' => 'SVMModel',
        'aliasName' => NULL,
      ),
      'load' => 
      array (
        'name' => 'load',
        'parameters' => 
        array (
          'filename' => 
          array (
            'name' => 'filename',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'string',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 79,
            'endLine' => 79,
            'startColumn' => 30,
            'endColumn' => 45,
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
 * Load a saved SVM Model
 * @param string $filename The filename of the model.
 * @return bool Returns true on success.
 * @throws SVMException
 * @link https://www.php.net/manual/en/svmmodel.load.php
 */',
        'startLine' => 79,
        'endLine' => 81,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'SVMModel',
        'implementingClassName' => 'SVMModel',
        'currentClassName' => 'SVMModel',
        'aliasName' => NULL,
      ),
      'predict_probability' => 
      array (
        'name' => 'predict_probability',
        'parameters' => 
        array (
          'data' => 
          array (
            'name' => 'data',
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
            'startLine' => 91,
            'endLine' => 91,
            'startColumn' => 45,
            'endColumn' => 55,
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
            'name' => 'float',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return class probabilities for previous unseen data
 *
 * This function accepts an array of data and attempts to predict the class, as with the predict function. Additionally, however, this function returns an array of probabilities, one per class in the model, which represent the estimated chance of the data supplied being a member of that class. Requires that the model to be used has been trained with the probability parameter set to true.
 * @param array $data The array to be classified. This should be a series of key => value pairs in increasing key order, but not necessarily continuous.
 * @return float the predicted value. This will be a class label in the case of classification, a real value in the case of regression. Throws SVMException on error
 * @throws SVMException Throws SVMException on error
 * @link https://www.php.net/manual/en/svmmodel.predict-probability.php
 */',
        'startLine' => 91,
        'endLine' => 93,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'SVMModel',
        'implementingClassName' => 'SVMModel',
        'currentClassName' => 'SVMModel',
        'aliasName' => NULL,
      ),
      'predict' => 
      array (
        'name' => 'predict',
        'parameters' => 
        array (
          'data' => 
          array (
            'name' => 'data',
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
            'startLine' => 103,
            'endLine' => 103,
            'startColumn' => 33,
            'endColumn' => 43,
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
            'name' => 'float',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Predict a value for previously unseen data
 *
 * This function accepts an array of data and attempts to predict the class or regression value based on the model extracted from previously trained data.
 * @param array $data The array to be classified. This should be a series of key => value pairs in increasing key order, but not necessarily continuous.
 * @return float the predicted value. This will be a class label in the case of classification, a real value in the case of regression. Throws SVMException on error
 * @throws SVMException Throws SVMException on error
 * @link https://www.php.net/manual/en/svmmodel.predict.php
 */',
        'startLine' => 103,
        'endLine' => 105,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'SVMModel',
        'implementingClassName' => 'SVMModel',
        'currentClassName' => 'SVMModel',
        'aliasName' => NULL,
      ),
      'save' => 
      array (
        'name' => 'save',
        'parameters' => 
        array (
          'filename' => 
          array (
            'name' => 'filename',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'string',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 113,
            'endLine' => 113,
            'startColumn' => 30,
            'endColumn' => 45,
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
 * Save a model to a file, for later use
 * @param string $filename The file to save the model to.
 * @return bool Throws SVMException on error. Returns true on success.
 * @throws SVMException Throws SVMException on error
 * @link https://www.php.net/manual/en/svmmodel.save.php
 */',
        'startLine' => 113,
        'endLine' => 115,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'SVMModel',
        'implementingClassName' => 'SVMModel',
        'currentClassName' => 'SVMModel',
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