<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/benchmarks/AnomalyDetectors/IsolationForestBench.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Benchmarks\AnomalyDetectors\IsolationForestBench
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-40e882d05188ac5be5158623a458b8c44697329b08dc3eea582a586903f21bf8',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Benchmarks\\AnomalyDetectors\\IsolationForestBench',
        'filename' => '/home/andrew/Workspace/Rubix/ML/benchmarks/AnomalyDetectors/IsolationForestBench.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Benchmarks\\AnomalyDetectors',
    'name' => 'Rubix\\ML\\Benchmarks\\AnomalyDetectors\\IsolationForestBench',
    'shortName' => 'IsolationForestBench',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * @Groups({"AnomalyDetectors"})
 * @BeforeMethods({"setUp"})
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 14,
    'endLine' => 51,
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
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\AnomalyDetectors\\IsolationForestBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\AnomalyDetectors\\IsolationForestBench',
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
            'startLine' => 16,
            'endLine' => 16,
            'startTokenPos' => 45,
            'startFilePos' => 359,
            'endTokenPos' => 45,
            'endFilePos' => 363,
          ),
        ),
        'docComment' => NULL,
        'attributes' => 
        array (
        ),
        'startLine' => 16,
        'endLine' => 16,
        'startColumn' => 5,
        'endColumn' => 46,
      ),
      'TESTING_SIZE' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\AnomalyDetectors\\IsolationForestBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\AnomalyDetectors\\IsolationForestBench',
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
            'startLine' => 18,
            'endLine' => 18,
            'startTokenPos' => 58,
            'startFilePos' => 406,
            'endTokenPos' => 58,
            'endFilePos' => 410,
          ),
        ),
        'docComment' => NULL,
        'attributes' => 
        array (
        ),
        'startLine' => 18,
        'endLine' => 18,
        'startColumn' => 5,
        'endColumn' => 45,
      ),
    ),
    'immediateProperties' => 
    array (
      'training' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\AnomalyDetectors\\IsolationForestBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\AnomalyDetectors\\IsolationForestBench',
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
        'startLine' => 20,
        'endLine' => 20,
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
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\AnomalyDetectors\\IsolationForestBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\AnomalyDetectors\\IsolationForestBench',
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
        'startLine' => 22,
        'endLine' => 22,
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
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\AnomalyDetectors\\IsolationForestBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\AnomalyDetectors\\IsolationForestBench',
        'name' => 'estimator',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Rubix\\ML\\AnomalyDetectors\\IsolationForest',
            'isIdentifier' => false,
          ),
        ),
        'default' => NULL,
        'docComment' => NULL,
        'attributes' => 
        array (
        ),
        'startLine' => 24,
        'endLine' => 24,
        'startColumn' => 5,
        'endColumn' => 41,
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
        'startLine' => 26,
        'endLine' => 38,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Benchmarks\\AnomalyDetectors',
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\AnomalyDetectors\\IsolationForestBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\AnomalyDetectors\\IsolationForestBench',
        'currentClassName' => 'Rubix\\ML\\Benchmarks\\AnomalyDetectors\\IsolationForestBench',
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
        'startLine' => 45,
        'endLine' => 50,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Benchmarks\\AnomalyDetectors',
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\AnomalyDetectors\\IsolationForestBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\AnomalyDetectors\\IsolationForestBench',
        'currentClassName' => 'Rubix\\ML\\Benchmarks\\AnomalyDetectors\\IsolationForestBench',
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