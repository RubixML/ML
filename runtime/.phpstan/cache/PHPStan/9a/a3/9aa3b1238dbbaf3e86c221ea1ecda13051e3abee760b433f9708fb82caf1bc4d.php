<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/benchmarks/Graph/Trees/BallTreeBench.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Benchmarks\Graph\Trees\BallTreeBench
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-69a8a5885ae74fcc62f9b27c0713ff5517a1235ce365653e8cd4d9d69f7fce16',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Benchmarks\\Graph\\Trees\\BallTreeBench',
        'filename' => '/home/andrew/Workspace/Rubix/ML/benchmarks/Graph/Trees/BallTreeBench.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Benchmarks\\Graph\\Trees',
    'name' => 'Rubix\\ML\\Benchmarks\\Graph\\Trees\\BallTreeBench',
    'shortName' => 'BallTreeBench',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * @Groups({"Trees"})
 * @BeforeMethods({"setUp"})
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 13,
    'endLine' => 49,
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
      'DATASET_SIZE' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Graph\\Trees\\BallTreeBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Graph\\Trees\\BallTreeBench',
        'name' => 'DATASET_SIZE',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '10000',
          'attributes' => 
          array (
            'startLine' => 15,
            'endLine' => 15,
            'startTokenPos' => 38,
            'startFilePos' => 288,
            'endTokenPos' => 38,
            'endFilePos' => 292,
          ),
        ),
        'docComment' => NULL,
        'attributes' => 
        array (
        ),
        'startLine' => 15,
        'endLine' => 15,
        'startColumn' => 5,
        'endColumn' => 41,
      ),
    ),
    'immediateProperties' => 
    array (
      'dataset' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Graph\\Trees\\BallTreeBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Graph\\Trees\\BallTreeBench',
        'name' => 'dataset',
        'modifiers' => 2,
        'type' => NULL,
        'default' => NULL,
        'docComment' => '/**
 * @var \\Rubix\\ML\\Datasets\\Labeled
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 20,
        'endLine' => 20,
        'startColumn' => 5,
        'endColumn' => 23,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'tree' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Graph\\Trees\\BallTreeBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Graph\\Trees\\BallTreeBench',
        'name' => 'tree',
        'modifiers' => 2,
        'type' => NULL,
        'default' => NULL,
        'docComment' => '/**
 * @var BallTree
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 25,
        'endLine' => 25,
        'startColumn' => 5,
        'endColumn' => 20,
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
        'startLine' => 27,
        'endLine' => 38,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Benchmarks\\Graph\\Trees',
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Graph\\Trees\\BallTreeBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Graph\\Trees\\BallTreeBench',
        'currentClassName' => 'Rubix\\ML\\Benchmarks\\Graph\\Trees\\BallTreeBench',
        'aliasName' => NULL,
      ),
      'grow' => 
      array (
        'name' => 'grow',
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
 * @Iterations(3)
 * @OutputTimeUnit("seconds", precision=3)
 */',
        'startLine' => 45,
        'endLine' => 48,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Benchmarks\\Graph\\Trees',
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Graph\\Trees\\BallTreeBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Graph\\Trees\\BallTreeBench',
        'currentClassName' => 'Rubix\\ML\\Benchmarks\\Graph\\Trees\\BallTreeBench',
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