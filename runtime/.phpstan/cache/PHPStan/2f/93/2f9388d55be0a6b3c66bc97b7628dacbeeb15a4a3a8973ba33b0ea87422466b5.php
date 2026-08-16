<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/benchmarks/Transformers/StopWordFilterBench.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Benchmarks\Transformers\StopWordFilterBench
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-21e44dd42f89bed2840d57b6c5e7d28ac956e1416d956810288568e1392f227f',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Benchmarks\\Transformers\\StopWordFilterBench',
        'filename' => '/home/andrew/Workspace/Rubix/ML/benchmarks/Transformers/StopWordFilterBench.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Benchmarks\\Transformers',
    'name' => 'Rubix\\ML\\Benchmarks\\Transformers\\StopWordFilterBench',
    'shortName' => 'StopWordFilterBench',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * @Groups({"Transformers"})
 * @BeforeMethods({"setUp"})
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 12,
    'endLine' => 62,
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
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Transformers\\StopWordFilterBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Transformers\\StopWordFilterBench',
        'name' => 'DATASET_SIZE',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '10000',
          'attributes' => 
          array (
            'startLine' => 14,
            'endLine' => 14,
            'startTokenPos' => 33,
            'startFilePos' => 257,
            'endTokenPos' => 33,
            'endFilePos' => 261,
          ),
        ),
        'docComment' => NULL,
        'attributes' => 
        array (
        ),
        'startLine' => 14,
        'endLine' => 14,
        'startColumn' => 5,
        'endColumn' => 41,
      ),
      'STOP_WORDS' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Transformers\\StopWordFilterBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Transformers\\StopWordFilterBench',
        'name' => 'STOP_WORDS',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '[\'a\', \'an\', \'and\', \'are\', \'as\', \'at\', \'be\', \'but\', \'by\', \'for\', \'if\', \'in\', \'into\', \'is\', \'it\', \'no\', \'not\', \'of\', \'on\', \'or\', \'such\', \'that\', \'the\', \'their\', \'then\', \'there\', \'these\', \'they\', \'this\', \'to\', \'was\', \'will\', \'with\']',
          'attributes' => 
          array (
            'startLine' => 16,
            'endLine' => 20,
            'startTokenPos' => 44,
            'startFilePos' => 298,
            'endTokenPos' => 145,
            'endFilePos' => 557,
          ),
        ),
        'docComment' => NULL,
        'attributes' => 
        array (
        ),
        'startLine' => 16,
        'endLine' => 20,
        'startColumn' => 5,
        'endColumn' => 6,
      ),
      'SAMPLE_TEXT' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Transformers\\StopWordFilterBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Transformers\\StopWordFilterBench',
        'name' => 'SAMPLE_TEXT',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '"Machine learning is the study of computer algorithms that improve automatically through experience. It is seen as a subset of artificial intelligence. Machine learning algorithms build a mathematical model based on sample data, known as \'training data\', in order to make predictions or decisions without being explicitly programmed to do so. Machine learning algorithms are used in a wide variety of applications, such as email filtering and computer vision, where it is difficult or infeasible to develop conventional algorithms to perform the needed tasks. Machine learning is closely related to computational statistics, which focuses on making predictions using computers. The study of mathematical optimization delivers methods, theory and application domains to the field of machine learning. Data mining is a related field of study, focusing on exploratory data analysis through unsupervised learning. In its application across business problems, machine learning is also referred to as predictive analytics."',
          'attributes' => 
          array (
            'startLine' => 22,
            'endLine' => 22,
            'startTokenPos' => 156,
            'startFilePos' => 595,
            'endTokenPos' => 156,
            'endFilePos' => 1611,
          ),
        ),
        'docComment' => NULL,
        'attributes' => 
        array (
        ),
        'startLine' => 22,
        'endLine' => 22,
        'startColumn' => 5,
        'endColumn' => 1052,
      ),
    ),
    'immediateProperties' => 
    array (
      'dataset' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Transformers\\StopWordFilterBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Transformers\\StopWordFilterBench',
        'name' => 'dataset',
        'modifiers' => 2,
        'type' => NULL,
        'default' => NULL,
        'docComment' => '/**
 * @var Unlabeled
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 27,
        'endLine' => 27,
        'startColumn' => 5,
        'endColumn' => 23,
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
        'startLine' => 29,
        'endLine' => 41,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Benchmarks\\Transformers',
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Transformers\\StopWordFilterBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Transformers\\StopWordFilterBench',
        'currentClassName' => 'Rubix\\ML\\Benchmarks\\Transformers\\StopWordFilterBench',
        'aliasName' => NULL,
      ),
      'apply' => 
      array (
        'name' => 'apply',
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
 * @OutputTimeUnit("milliseconds", precision=3)
 */',
        'startLine' => 48,
        'endLine' => 51,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Benchmarks\\Transformers',
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Transformers\\StopWordFilterBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Transformers\\StopWordFilterBench',
        'currentClassName' => 'Rubix\\ML\\Benchmarks\\Transformers\\StopWordFilterBench',
        'aliasName' => NULL,
      ),
      'applyEmpty' => 
      array (
        'name' => 'applyEmpty',
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
 * @OutputTimeUnit("milliseconds", precision=3)
 */',
        'startLine' => 58,
        'endLine' => 61,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Benchmarks\\Transformers',
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Transformers\\StopWordFilterBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Transformers\\StopWordFilterBench',
        'currentClassName' => 'Rubix\\ML\\Benchmarks\\Transformers\\StopWordFilterBench',
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