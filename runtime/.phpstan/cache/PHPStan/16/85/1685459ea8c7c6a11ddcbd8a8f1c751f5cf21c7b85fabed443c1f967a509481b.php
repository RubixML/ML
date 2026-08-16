<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/benchmarks/Regressors/RadiusNeighborsRegressorBench.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Benchmarks\Regressors\RadiusNeighborsRegressorBench
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-d2a47307b1180cdfa5cbd38d9a325b0ab9bf9fcb9f65876df0234a56f292f6ca',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Benchmarks\\Regressors\\RadiusNeighborsRegressorBench',
        'filename' => '/home/andrew/Workspace/Rubix/ML/benchmarks/Regressors/RadiusNeighborsRegressorBench.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Benchmarks\\Regressors',
    'name' => 'Rubix\\ML\\Benchmarks\\Regressors\\RadiusNeighborsRegressorBench',
    'shortName' => 'RadiusNeighborsRegressorBench',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * @Groups({"Regressors"})
 * @BeforeMethods({"setUp"})
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 13,
    'endLine' => 47,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
    ),
    'traitClassNames' => 
    array (
    ),
    'immediateConstants' => 
    array (
      'TRAINING_SIZE' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Regressors\\RadiusNeighborsRegressorBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Regressors\\RadiusNeighborsRegressorBench',
        'name' => 'TRAINING_SIZE',
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
        'value' => 
        array (
          'code' => '10000',
          'attributes' => 
          array (
            'startLine' => 15,
            'endLine' => 15,
            'startTokenPos' => 40,
            'startFilePos' => 319,
            'endTokenPos' => 40,
            'endFilePos' => 323,
          ),
        ),
        'docComment' => NULL,
        'attributes' => 
        array (
        ),
        'startLine' => 15,
        'endLine' => 15,
        'startColumn' => 5,
        'endColumn' => 46,
      ),
      'TESTING_SIZE' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Regressors\\RadiusNeighborsRegressorBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Regressors\\RadiusNeighborsRegressorBench',
        'name' => 'TESTING_SIZE',
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
        'value' => 
        array (
          'code' => '10000',
          'attributes' => 
          array (
            'startLine' => 17,
            'endLine' => 17,
            'startTokenPos' => 53,
            'startFilePos' => 366,
            'endTokenPos' => 53,
            'endFilePos' => 370,
          ),
        ),
        'docComment' => NULL,
        'attributes' => 
        array (
        ),
        'startLine' => 17,
        'endLine' => 17,
        'startColumn' => 5,
        'endColumn' => 45,
      ),
    ),
    'immediateProperties' => 
    array (
      'training' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Regressors\\RadiusNeighborsRegressorBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Regressors\\RadiusNeighborsRegressorBench',
        'name' => 'training',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Rubix\\ML\\Datasets\\Labeled',
            'isIdentifier' => false,
          ),
        ),
        'default' => NULL,
        'docComment' => NULL,
        'attributes' => 
        array (
        ),
        'startLine' => 19,
        'endLine' => 19,
        'startColumn' => 5,
        'endColumn' => 32,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'testing' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Regressors\\RadiusNeighborsRegressorBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Regressors\\RadiusNeighborsRegressorBench',
        'name' => 'testing',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Rubix\\ML\\Datasets\\Labeled',
            'isIdentifier' => false,
          ),
        ),
        'default' => NULL,
        'docComment' => NULL,
        'attributes' => 
        array (
        ),
        'startLine' => 21,
        'endLine' => 21,
        'startColumn' => 5,
        'endColumn' => 31,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'estimator' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Regressors\\RadiusNeighborsRegressorBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Regressors\\RadiusNeighborsRegressorBench',
        'name' => 'estimator',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Rubix\\ML\\Regressors\\RadiusNeighborsRegressor',
            'isIdentifier' => false,
          ),
        ),
        'default' => NULL,
        'docComment' => NULL,
        'attributes' => 
        array (
        ),
        'startLine' => 23,
        'endLine' => 23,
        'startColumn' => 5,
        'endColumn' => 50,
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
      'setUp' => 
      array (
        'name' => 'setUp',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'void',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => NULL,
        'startLine' => 25,
        'endLine' => 34,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Benchmarks\\Regressors',
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Regressors\\RadiusNeighborsRegressorBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Regressors\\RadiusNeighborsRegressorBench',
        'currentClassName' => 'Rubix\\ML\\Benchmarks\\Regressors\\RadiusNeighborsRegressorBench',
        'aliasName' => NULL,
      ),
      'trainPredict' => 
      array (
        'name' => 'trainPredict',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'void',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * @Subject
 * @Iterations(5)
 * @OutputTimeUnit("seconds", precision=3)
 */',
        'startLine' => 41,
        'endLine' => 46,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Benchmarks\\Regressors',
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Regressors\\RadiusNeighborsRegressorBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Regressors\\RadiusNeighborsRegressorBench',
        'currentClassName' => 'Rubix\\ML\\Benchmarks\\Regressors\\RadiusNeighborsRegressorBench',
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