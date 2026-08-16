<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/CrossValidation/MonteCarlo.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\CrossValidation\MonteCarlo
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-5c5a04c39932743ef5a22c9f4620cbce8fa40214b6a6ce85d1222934511fe92d',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\CrossValidation\\MonteCarlo',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/CrossValidation/MonteCarlo.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\CrossValidation',
    'name' => 'Rubix\\ML\\CrossValidation\\MonteCarlo',
    'shortName' => 'MonteCarlo',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Monte Carlo
 *
 * Monte Carlo cross validation (or *repeated random subsampling*) is a technique that
 * averages the validation score of a learner over a user-defined number of simulations
 * where the learner is trained and tested on random splits of the dataset. The estimated
 * validation score approaches the actual validation score as the number of simulations
 * goes to infinity, however, only a tiny fraction of all possible simulations are needed
 * to produce a pretty good approximation.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 33,
    'endLine' => 124,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Rubix\\ML\\CrossValidation\\Validator',
      1 => 'Rubix\\ML\\Parallel',
    ),
    'traitClassNames' => 
    array (
      0 => 'Rubix\\ML\\Traits\\Multiprocessing',
    ),
    'immediateConstants' => 
    array (
    ),
    'immediateProperties' => 
    array (
      'simulations' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\CrossValidation\\MonteCarlo',
        'implementingClassName' => 'Rubix\\ML\\CrossValidation\\MonteCarlo',
        'name' => 'simulations',
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
 * The number of simulations i.e. random subsamplings of the dataset.
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
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'ratio' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\CrossValidation\\MonteCarlo',
        'implementingClassName' => 'Rubix\\ML\\CrossValidation\\MonteCarlo',
        'name' => 'ratio',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'float',
            'isIdentifier' => true,
          ),
        ),
        'default' => NULL,
        'docComment' => '/**
 * The hold out ratio. i.e. the ratio of samples to use for testing.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 49,
        'endLine' => 49,
        'startColumn' => 5,
        'endColumn' => 27,
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
      '__construct' => 
      array (
        'name' => '__construct',
        'parameters' => 
        array (
          'simulations' => 
          array (
            'name' => 'simulations',
            'default' => 
            array (
              'code' => '10',
              'attributes' => 
              array (
                'startLine' => 56,
                'endLine' => 56,
                'startTokenPos' => 124,
                'startFilePos' => 1649,
                'endTokenPos' => 124,
                'endFilePos' => 1650,
              ),
            ),
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
            'startLine' => 56,
            'endLine' => 56,
            'startColumn' => 33,
            'endColumn' => 53,
            'parameterIndex' => 0,
            'isOptional' => true,
          ),
          'ratio' => 
          array (
            'name' => 'ratio',
            'default' => 
            array (
              'code' => '0.2',
              'attributes' => 
              array (
                'startLine' => 56,
                'endLine' => 56,
                'startTokenPos' => 133,
                'startFilePos' => 1668,
                'endTokenPos' => 133,
                'endFilePos' => 1670,
              ),
            ),
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'float',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 56,
            'endLine' => 56,
            'startColumn' => 56,
            'endColumn' => 73,
            'parameterIndex' => 1,
            'isOptional' => true,
          ),
        ),
        'returnsReference' => false,
        'returnType' => NULL,
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * @param int $simulations
 * @param float $ratio
 * @throws InvalidArgumentException
 */',
        'startLine' => 56,
        'endLine' => 71,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\CrossValidation',
        'declaringClassName' => 'Rubix\\ML\\CrossValidation\\MonteCarlo',
        'implementingClassName' => 'Rubix\\ML\\CrossValidation\\MonteCarlo',
        'currentClassName' => 'Rubix\\ML\\CrossValidation\\MonteCarlo',
        'aliasName' => NULL,
      ),
      'test' => 
      array (
        'name' => 'test',
        'parameters' => 
        array (
          'estimator' => 
          array (
            'name' => 'estimator',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Rubix\\ML\\Learner',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 82,
            'endLine' => 82,
            'startColumn' => 26,
            'endColumn' => 43,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'dataset' => 
          array (
            'name' => 'dataset',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Rubix\\ML\\Datasets\\Labeled',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 82,
            'endLine' => 82,
            'startColumn' => 46,
            'endColumn' => 61,
            'parameterIndex' => 1,
            'isOptional' => false,
          ),
          'metric' => 
          array (
            'name' => 'metric',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Rubix\\ML\\CrossValidation\\Metrics\\Metric',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 82,
            'endLine' => 82,
            'startColumn' => 64,
            'endColumn' => 77,
            'parameterIndex' => 2,
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
 * Test the estimator with the supplied dataset and return a validation score.
 *
 * @param Learner $estimator
 * @param Labeled $dataset
 * @param Metric $metric
 * @throws RuntimeException
 * @return float
 */',
        'startLine' => 82,
        'endLine' => 111,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\CrossValidation',
        'declaringClassName' => 'Rubix\\ML\\CrossValidation\\MonteCarlo',
        'implementingClassName' => 'Rubix\\ML\\CrossValidation\\MonteCarlo',
        'currentClassName' => 'Rubix\\ML\\CrossValidation\\MonteCarlo',
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
 * Return the string representation of the object.
 *
 * @internal
 *
 * @return string
 */',
        'startLine' => 120,
        'endLine' => 123,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\CrossValidation',
        'declaringClassName' => 'Rubix\\ML\\CrossValidation\\MonteCarlo',
        'implementingClassName' => 'Rubix\\ML\\CrossValidation\\MonteCarlo',
        'currentClassName' => 'Rubix\\ML\\CrossValidation\\MonteCarlo',
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