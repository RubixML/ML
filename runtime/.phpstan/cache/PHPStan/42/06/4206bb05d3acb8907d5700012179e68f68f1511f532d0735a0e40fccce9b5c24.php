<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/EstimatorType.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\EstimatorType
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-d9511d18965db5ec68dbc868143b61c7c18595683e8ff113c8c9cc431db160b6',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\EstimatorType',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/EstimatorType.php',
      ),
    ),
    'namespace' => 'Rubix\\ML',
    'name' => 'Rubix\\ML\\EstimatorType',
    'shortName' => 'EstimatorType',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Estimator Type
 *
 * Estimator type enum.
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 21,
    'endLine' => 212,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Stringable',
    ),
    'traitClassNames' => 
    array (
    ),
    'immediateConstants' => 
    array (
      'CLASSIFIER' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\EstimatorType',
        'implementingClassName' => 'Rubix\\ML\\EstimatorType',
        'name' => 'CLASSIFIER',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '1',
          'attributes' => 
          array (
            'startLine' => 28,
            'endLine' => 28,
            'startTokenPos' => 46,
            'startFilePos' => 436,
            'endTokenPos' => 46,
            'endFilePos' => 436,
          ),
        ),
        'docComment' => '/**
 * The classifier estimator type code.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 28,
        'endLine' => 28,
        'startColumn' => 5,
        'endColumn' => 32,
      ),
      'REGRESSOR' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\EstimatorType',
        'implementingClassName' => 'Rubix\\ML\\EstimatorType',
        'name' => 'REGRESSOR',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '2',
          'attributes' => 
          array (
            'startLine' => 35,
            'endLine' => 35,
            'startTokenPos' => 59,
            'startFilePos' => 550,
            'endTokenPos' => 59,
            'endFilePos' => 550,
          ),
        ),
        'docComment' => '/**
 * The regressor estimator type code.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 35,
        'endLine' => 35,
        'startColumn' => 5,
        'endColumn' => 31,
      ),
      'CLUSTERER' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\EstimatorType',
        'implementingClassName' => 'Rubix\\ML\\EstimatorType',
        'name' => 'CLUSTERER',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '3',
          'attributes' => 
          array (
            'startLine' => 42,
            'endLine' => 42,
            'startTokenPos' => 72,
            'startFilePos' => 664,
            'endTokenPos' => 72,
            'endFilePos' => 664,
          ),
        ),
        'docComment' => '/**
 * The clusterer estimator type code.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 42,
        'endLine' => 42,
        'startColumn' => 5,
        'endColumn' => 31,
      ),
      'ANOMALY_DETECTOR' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\EstimatorType',
        'implementingClassName' => 'Rubix\\ML\\EstimatorType',
        'name' => 'ANOMALY_DETECTOR',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '4',
          'attributes' => 
          array (
            'startLine' => 49,
            'endLine' => 49,
            'startTokenPos' => 85,
            'startFilePos' => 792,
            'endTokenPos' => 85,
            'endFilePos' => 792,
          ),
        ),
        'docComment' => '/**
 * The anomaly detector estimator type code.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 49,
        'endLine' => 49,
        'startColumn' => 5,
        'endColumn' => 38,
      ),
      'TYPE_STRINGS' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\EstimatorType',
        'implementingClassName' => 'Rubix\\ML\\EstimatorType',
        'name' => 'TYPE_STRINGS',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '[self::CLASSIFIER => \'classifier\', self::REGRESSOR => \'regressor\', self::CLUSTERER => \'clusterer\', self::ANOMALY_DETECTOR => \'anomaly detector\']',
          'attributes' => 
          array (
            'startLine' => 56,
            'endLine' => 61,
            'startTokenPos' => 98,
            'startFilePos' => 964,
            'endTokenPos' => 136,
            'endFilePos' => 1146,
          ),
        ),
        'docComment' => '/**
 * An array of human-readable string representations of the estimator types.
 *
 * @var literal-string[]
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 56,
        'endLine' => 61,
        'startColumn' => 5,
        'endColumn' => 6,
      ),
      'ALL' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\EstimatorType',
        'implementingClassName' => 'Rubix\\ML\\EstimatorType',
        'name' => 'ALL',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '[self::CLASSIFIER, self::REGRESSOR, self::CLUSTERER, self::ANOMALY_DETECTOR]',
          'attributes' => 
          array (
            'startLine' => 68,
            'endLine' => 73,
            'startTokenPos' => 149,
            'startFilePos' => 1270,
            'endTokenPos' => 171,
            'endFilePos' => 1384,
          ),
        ),
        'docComment' => '/**
 * An array of all the estimator type codes.
 *
 * @var list<int>
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 68,
        'endLine' => 73,
        'startColumn' => 5,
        'endColumn' => 6,
      ),
    ),
    'immediateProperties' => 
    array (
      'code' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\EstimatorType',
        'implementingClassName' => 'Rubix\\ML\\EstimatorType',
        'name' => 'code',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'int',
            'isIdentifier' => true,
          ),
        ),
        'default' => NULL,
        'docComment' => '/**
 * The integer-encoded estimator type.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 80,
        'endLine' => 80,
        'startColumn' => 5,
        'endColumn' => 24,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
    ),
    'immediateMethods' => 
    array (
      'build' => 
      array (
        'name' => 'build',
        'parameters' => 
        array (
          'code' => 
          array (
            'name' => 'code',
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
            'startLine' => 87,
            'endLine' => 87,
            'startColumn' => 34,
            'endColumn' => 42,
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
            'name' => 'self',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Build a new estimator type object.
 *
 * @param int $code
 */',
        'startLine' => 87,
        'endLine' => 90,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 17,
        'namespace' => 'Rubix\\ML',
        'declaringClassName' => 'Rubix\\ML\\EstimatorType',
        'implementingClassName' => 'Rubix\\ML\\EstimatorType',
        'currentClassName' => 'Rubix\\ML\\EstimatorType',
        'aliasName' => NULL,
      ),
      'classifier' => 
      array (
        'name' => 'classifier',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'self',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Build a classifier type.
 *
 * @return self
 */',
        'startLine' => 97,
        'endLine' => 100,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 17,
        'namespace' => 'Rubix\\ML',
        'declaringClassName' => 'Rubix\\ML\\EstimatorType',
        'implementingClassName' => 'Rubix\\ML\\EstimatorType',
        'currentClassName' => 'Rubix\\ML\\EstimatorType',
        'aliasName' => NULL,
      ),
      'regressor' => 
      array (
        'name' => 'regressor',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'self',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Build a regressor type.
 *
 * @return self
 */',
        'startLine' => 107,
        'endLine' => 110,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 17,
        'namespace' => 'Rubix\\ML',
        'declaringClassName' => 'Rubix\\ML\\EstimatorType',
        'implementingClassName' => 'Rubix\\ML\\EstimatorType',
        'currentClassName' => 'Rubix\\ML\\EstimatorType',
        'aliasName' => NULL,
      ),
      'clusterer' => 
      array (
        'name' => 'clusterer',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'self',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Build a clusterer type.
 *
 * @return self
 */',
        'startLine' => 117,
        'endLine' => 120,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 17,
        'namespace' => 'Rubix\\ML',
        'declaringClassName' => 'Rubix\\ML\\EstimatorType',
        'implementingClassName' => 'Rubix\\ML\\EstimatorType',
        'currentClassName' => 'Rubix\\ML\\EstimatorType',
        'aliasName' => NULL,
      ),
      'anomalyDetector' => 
      array (
        'name' => 'anomalyDetector',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'self',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Build an anomaly detector type.
 *
 * @return self
 */',
        'startLine' => 127,
        'endLine' => 130,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 17,
        'namespace' => 'Rubix\\ML',
        'declaringClassName' => 'Rubix\\ML\\EstimatorType',
        'implementingClassName' => 'Rubix\\ML\\EstimatorType',
        'currentClassName' => 'Rubix\\ML\\EstimatorType',
        'aliasName' => NULL,
      ),
      '__construct' => 
      array (
        'name' => '__construct',
        'parameters' => 
        array (
          'code' => 
          array (
            'name' => 'code',
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
            'startLine' => 136,
            'endLine' => 136,
            'startColumn' => 33,
            'endColumn' => 41,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
        ),
        'returnsReference' => false,
        'returnType' => NULL,
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * @param int $code
 * @throws InvalidArgumentException
 */',
        'startLine' => 136,
        'endLine' => 143,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML',
        'declaringClassName' => 'Rubix\\ML\\EstimatorType',
        'implementingClassName' => 'Rubix\\ML\\EstimatorType',
        'currentClassName' => 'Rubix\\ML\\EstimatorType',
        'aliasName' => NULL,
      ),
      'code' => 
      array (
        'name' => 'code',
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
 * Return the integer-encoded estimator type.
 *
 * @return int
 */',
        'startLine' => 150,
        'endLine' => 153,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML',
        'declaringClassName' => 'Rubix\\ML\\EstimatorType',
        'implementingClassName' => 'Rubix\\ML\\EstimatorType',
        'currentClassName' => 'Rubix\\ML\\EstimatorType',
        'aliasName' => NULL,
      ),
      'isSupervised' => 
      array (
        'name' => 'isSupervised',
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
 * Is the estimator type supervised?
 */',
        'startLine' => 158,
        'endLine' => 161,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML',
        'declaringClassName' => 'Rubix\\ML\\EstimatorType',
        'implementingClassName' => 'Rubix\\ML\\EstimatorType',
        'currentClassName' => 'Rubix\\ML\\EstimatorType',
        'aliasName' => NULL,
      ),
      'isClassifier' => 
      array (
        'name' => 'isClassifier',
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
 * Is it a classifier?
 *
 * @return bool
 */',
        'startLine' => 168,
        'endLine' => 171,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML',
        'declaringClassName' => 'Rubix\\ML\\EstimatorType',
        'implementingClassName' => 'Rubix\\ML\\EstimatorType',
        'currentClassName' => 'Rubix\\ML\\EstimatorType',
        'aliasName' => NULL,
      ),
      'isRegressor' => 
      array (
        'name' => 'isRegressor',
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
 * Is it a regressor?
 *
 * @return bool
 */',
        'startLine' => 178,
        'endLine' => 181,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML',
        'declaringClassName' => 'Rubix\\ML\\EstimatorType',
        'implementingClassName' => 'Rubix\\ML\\EstimatorType',
        'currentClassName' => 'Rubix\\ML\\EstimatorType',
        'aliasName' => NULL,
      ),
      'isClusterer' => 
      array (
        'name' => 'isClusterer',
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
 * Is it a clusterer?
 *
 * @return bool
 */',
        'startLine' => 188,
        'endLine' => 191,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML',
        'declaringClassName' => 'Rubix\\ML\\EstimatorType',
        'implementingClassName' => 'Rubix\\ML\\EstimatorType',
        'currentClassName' => 'Rubix\\ML\\EstimatorType',
        'aliasName' => NULL,
      ),
      'isAnomalyDetector' => 
      array (
        'name' => 'isAnomalyDetector',
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
 * Is it an anomaly detector?
 *
 * @return bool
 */',
        'startLine' => 198,
        'endLine' => 201,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML',
        'declaringClassName' => 'Rubix\\ML\\EstimatorType',
        'implementingClassName' => 'Rubix\\ML\\EstimatorType',
        'currentClassName' => 'Rubix\\ML\\EstimatorType',
        'aliasName' => NULL,
      ),
      '__toString' => 
      array (
        'name' => '__toString',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'string',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the estimator type as a string.
 *
 * @return string
 */',
        'startLine' => 208,
        'endLine' => 211,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML',
        'declaringClassName' => 'Rubix\\ML\\EstimatorType',
        'implementingClassName' => 'Rubix\\ML\\EstimatorType',
        'currentClassName' => 'Rubix\\ML\\EstimatorType',
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