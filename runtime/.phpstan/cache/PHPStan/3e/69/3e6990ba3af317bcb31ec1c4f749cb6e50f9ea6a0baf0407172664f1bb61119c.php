<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Transformers/TSNE.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Transformers\TSNE
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-36c12e313ec2e043551e2d2d2b2087dace0224c2b4152a7e2be59cbf4370e308',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Transformers\\TSNE',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Transformers/TSNE.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Transformers',
    'name' => 'Rubix\\ML\\Transformers\\TSNE',
    'shortName' => 'TSNE',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * t-SNE
 *
 * *T-distributed Stochastic Neighbor Embedding* is a two-stage non-linear manifold
 * learning algorithm based on Batch Gradient Descent that seeks to maintain the
 * distances between samples in low-dimensional space. During the first stage (*early
 * stage*) the distances are exaggerated to encourage more pronounced clusters. Since
 * the t-SNE cost function (KL Divergence) has a rough gradient, momentum is employed
 * to help escape bad local minima.
 *
 * > **Note:** T-SNE is implemented using the *exact* method which scales quadratically
 * in the number of samples. Therefore, it is recommended to subsample datasets larger
 * than a few thousand samples.
 *
 * References:
 * [1] L. van der Maaten et al. (2008). Visualizing Data using t-SNE.
 * [2] L. van der Maaten. (2009). Learning a Parametric Embedding by Preserving
 * Local Structure.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 43,
    'endLine' => 581,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Rubix\\ML\\Transformers\\Transformer',
      1 => 'Rubix\\ML\\Verbose',
    ),
    'traitClassNames' => 
    array (
      0 => 'Rubix\\ML\\Traits\\LoggerAware',
    ),
    'immediateConstants' => 
    array (
      'MAX_EARLY_EPOCHS' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'name' => 'MAX_EARLY_EPOCHS',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '250',
          'attributes' => 
          array (
            'startLine' => 52,
            'endLine' => 52,
            'startTokenPos' => 101,
            'startFilePos' => 1614,
            'endTokenPos' => 101,
            'endFilePos' => 1616,
          ),
        ),
        'docComment' => '/**
 * The maximum number of epochs with early exaggeration.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 52,
        'endLine' => 52,
        'startColumn' => 5,
        'endColumn' => 43,
      ),
      'INIT_MOMENTUM' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'name' => 'INIT_MOMENTUM',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '0.5',
          'attributes' => 
          array (
            'startLine' => 59,
            'endLine' => 59,
            'startTokenPos' => 114,
            'startFilePos' => 1738,
            'endTokenPos' => 114,
            'endFilePos' => 1740,
          ),
        ),
        'docComment' => '/**
 * The initial momentum coefficient.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 59,
        'endLine' => 59,
        'startColumn' => 5,
        'endColumn' => 40,
      ),
      'MOMENTUM_BOOST' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'name' => 'MOMENTUM_BOOST',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '0.3',
          'attributes' => 
          array (
            'startLine' => 66,
            'endLine' => 66,
            'startTokenPos' => 127,
            'startFilePos' => 1894,
            'endTokenPos' => 127,
            'endFilePos' => 1896,
          ),
        ),
        'docComment' => '/**
 * The amount of momentum added after the early exaggeration stage.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 66,
        'endLine' => 66,
        'startColumn' => 5,
        'endColumn' => 41,
      ),
      'MAX_BINARY_PRECISION' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'name' => 'MAX_BINARY_PRECISION',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '100',
          'attributes' => 
          array (
            'startLine' => 73,
            'endLine' => 73,
            'startTokenPos' => 140,
            'startFilePos' => 2035,
            'endTokenPos' => 140,
            'endFilePos' => 2037,
          ),
        ),
        'docComment' => '/**
 * The maximum number of binary search attempts.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 73,
        'endLine' => 73,
        'startColumn' => 5,
        'endColumn' => 47,
      ),
      'PERPLEXITY_TOLERANCE' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'name' => 'PERPLEXITY_TOLERANCE',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '1.0E-5',
          'attributes' => 
          array (
            'startLine' => 80,
            'endLine' => 80,
            'startTokenPos' => 153,
            'startFilePos' => 2179,
            'endTokenPos' => 153,
            'endFilePos' => 2182,
          ),
        ),
        'docComment' => '/**
 * The amount of binary search error to tolerate.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 80,
        'endLine' => 80,
        'startColumn' => 5,
        'endColumn' => 48,
      ),
      'Y_INIT_SCALE' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'name' => 'Y_INIT_SCALE',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '0.0001',
          'attributes' => 
          array (
            'startLine' => 87,
            'endLine' => 87,
            'startTokenPos' => 166,
            'startFilePos' => 2319,
            'endTokenPos' => 166,
            'endFilePos' => 2322,
          ),
        ),
        'docComment' => '/**
 * The scaling coefficient of the initial embedding.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 87,
        'endLine' => 87,
        'startColumn' => 5,
        'endColumn' => 40,
      ),
      'GAIN_ACCELERATE' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'name' => 'GAIN_ACCELERATE',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '0.2',
          'attributes' => 
          array (
            'startLine' => 94,
            'endLine' => 94,
            'startTokenPos' => 179,
            'startFilePos' => 2487,
            'endTokenPos' => 179,
            'endFilePos' => 2489,
          ),
        ),
        'docComment' => '/**
 * The amount of gain to add while the direction of the gradient is the same.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 94,
        'endLine' => 94,
        'startColumn' => 5,
        'endColumn' => 42,
      ),
      'GAIN_BRAKE' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'name' => 'GAIN_BRAKE',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '0.8',
          'attributes' => 
          array (
            'startLine' => 101,
            'endLine' => 101,
            'startTokenPos' => 192,
            'startFilePos' => 2647,
            'endTokenPos' => 192,
            'endFilePos' => 2649,
          ),
        ),
        'docComment' => '/**
 * The amount of brake to apply when the direction of the gradient changes.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 101,
        'endLine' => 101,
        'startColumn' => 5,
        'endColumn' => 37,
      ),
      'MIN_GAIN' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'name' => 'MIN_GAIN',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '0.01',
          'attributes' => 
          array (
            'startLine' => 108,
            'endLine' => 108,
            'startTokenPos' => 205,
            'startFilePos' => 2784,
            'endTokenPos' => 205,
            'endFilePos' => 2787,
          ),
        ),
        'docComment' => '/**
 * The minimum amount of gain to apply at each update.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 108,
        'endLine' => 108,
        'startColumn' => 5,
        'endColumn' => 36,
      ),
    ),
    'immediateProperties' => 
    array (
      'dimensions' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'name' => 'dimensions',
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
 * The number of dimensions of the target embedding.
 *
 * @var positive-int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 115,
        'endLine' => 115,
        'startColumn' => 5,
        'endColumn' => 30,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'dofs' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'name' => 'dofs',
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
 * The number of degrees of freedom for the student\'s t distribution.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 122,
        'endLine' => 122,
        'startColumn' => 5,
        'endColumn' => 24,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'c' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'name' => 'c',
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
 * The precomputed c factor of the gradient computation.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 129,
        'endLine' => 129,
        'startColumn' => 5,
        'endColumn' => 23,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'rate' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'name' => 'rate',
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
 * The learning rate that controls the global step size.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 136,
        'endLine' => 136,
        'startColumn' => 5,
        'endColumn' => 26,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'perplexity' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'name' => 'perplexity',
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
 * The number of effective nearest neighbors to refer to when computing
 * the variance of the distribution over that sample.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 144,
        'endLine' => 144,
        'startColumn' => 5,
        'endColumn' => 32,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'entropy' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'name' => 'entropy',
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
 * The desired entropy of the Gaussian component over each sample i.e the log perplexity.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 151,
        'endLine' => 151,
        'startColumn' => 5,
        'endColumn' => 29,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'exaggeration' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'name' => 'exaggeration',
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
 * The factor to exaggerate the distances between samples by during the early stage of fitting.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 158,
        'endLine' => 158,
        'startColumn' => 5,
        'endColumn' => 34,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'epochs' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'name' => 'epochs',
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
 * The number of times to iterate over the embedding.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 165,
        'endLine' => 165,
        'startColumn' => 5,
        'endColumn' => 26,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'early' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'name' => 'early',
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
 * The number of epochs that are considered to be in the early training stage.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 172,
        'endLine' => 172,
        'startColumn' => 5,
        'endColumn' => 25,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'minGradient' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'name' => 'minGradient',
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
 * The minimum norm of the gradient necessary to continue embedding.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 179,
        'endLine' => 179,
        'startColumn' => 5,
        'endColumn' => 33,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'window' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'name' => 'window',
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
 * The number of epochs without improvement in the training loss to wait before considering an early stop.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 186,
        'endLine' => 186,
        'startColumn' => 5,
        'endColumn' => 26,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'kernel' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'name' => 'kernel',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Rubix\\ML\\Kernels\\Distance\\Distance',
            'isIdentifier' => false,
          ),
        ),
        'default' => NULL,
        'docComment' => '/**
 * The distance metric used to measure distances between samples in both high and low dimensions.
 *
 * @var Distance
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 193,
        'endLine' => 193,
        'startColumn' => 5,
        'endColumn' => 31,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'losses' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'name' => 'losses',
        'modifiers' => 2,
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
        'default' => 
        array (
          'code' => 'null',
          'attributes' => 
          array (
            'startLine' => 200,
            'endLine' => 200,
            'startTokenPos' => 327,
            'startFilePos' => 4790,
            'endTokenPos' => 327,
            'endFilePos' => 4793,
          ),
        ),
        'docComment' => '/**
 * The loss at each epoch from the last embedding.
 *
 * @var float[]|null
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 200,
        'endLine' => 200,
        'startColumn' => 5,
        'endColumn' => 36,
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
          'dimensions' => 
          array (
            'name' => 'dimensions',
            'default' => 
            array (
              'code' => '2',
              'attributes' => 
              array (
                'startLine' => 214,
                'endLine' => 214,
                'startTokenPos' => 345,
                'startFilePos' => 5153,
                'endTokenPos' => 345,
                'endFilePos' => 5153,
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
            'startLine' => 214,
            'endLine' => 214,
            'startColumn' => 9,
            'endColumn' => 27,
            'parameterIndex' => 0,
            'isOptional' => true,
          ),
          'rate' => 
          array (
            'name' => 'rate',
            'default' => 
            array (
              'code' => '100.0',
              'attributes' => 
              array (
                'startLine' => 215,
                'endLine' => 215,
                'startTokenPos' => 354,
                'startFilePos' => 5178,
                'endTokenPos' => 354,
                'endFilePos' => 5182,
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
            'startLine' => 215,
            'endLine' => 215,
            'startColumn' => 9,
            'endColumn' => 27,
            'parameterIndex' => 1,
            'isOptional' => true,
          ),
          'perplexity' => 
          array (
            'name' => 'perplexity',
            'default' => 
            array (
              'code' => '30',
              'attributes' => 
              array (
                'startLine' => 216,
                'endLine' => 216,
                'startTokenPos' => 363,
                'startFilePos' => 5211,
                'endTokenPos' => 363,
                'endFilePos' => 5212,
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
            'startLine' => 216,
            'endLine' => 216,
            'startColumn' => 9,
            'endColumn' => 28,
            'parameterIndex' => 2,
            'isOptional' => true,
          ),
          'exaggeration' => 
          array (
            'name' => 'exaggeration',
            'default' => 
            array (
              'code' => '12.0',
              'attributes' => 
              array (
                'startLine' => 217,
                'endLine' => 217,
                'startTokenPos' => 372,
                'startFilePos' => 5245,
                'endTokenPos' => 372,
                'endFilePos' => 5248,
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
            'startLine' => 217,
            'endLine' => 217,
            'startColumn' => 9,
            'endColumn' => 34,
            'parameterIndex' => 3,
            'isOptional' => true,
          ),
          'epochs' => 
          array (
            'name' => 'epochs',
            'default' => 
            array (
              'code' => '1000',
              'attributes' => 
              array (
                'startLine' => 218,
                'endLine' => 218,
                'startTokenPos' => 381,
                'startFilePos' => 5273,
                'endTokenPos' => 381,
                'endFilePos' => 5276,
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
            'startLine' => 218,
            'endLine' => 218,
            'startColumn' => 9,
            'endColumn' => 26,
            'parameterIndex' => 4,
            'isOptional' => true,
          ),
          'minGradient' => 
          array (
            'name' => 'minGradient',
            'default' => 
            array (
              'code' => '1.0E-7',
              'attributes' => 
              array (
                'startLine' => 219,
                'endLine' => 219,
                'startTokenPos' => 390,
                'startFilePos' => 5308,
                'endTokenPos' => 390,
                'endFilePos' => 5311,
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
            'startLine' => 219,
            'endLine' => 219,
            'startColumn' => 9,
            'endColumn' => 33,
            'parameterIndex' => 5,
            'isOptional' => true,
          ),
          'window' => 
          array (
            'name' => 'window',
            'default' => 
            array (
              'code' => '5',
              'attributes' => 
              array (
                'startLine' => 220,
                'endLine' => 220,
                'startTokenPos' => 399,
                'startFilePos' => 5336,
                'endTokenPos' => 399,
                'endFilePos' => 5336,
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
            'startLine' => 220,
            'endLine' => 220,
            'startColumn' => 9,
            'endColumn' => 23,
            'parameterIndex' => 6,
            'isOptional' => true,
          ),
          'kernel' => 
          array (
            'name' => 'kernel',
            'default' => 
            array (
              'code' => 'null',
              'attributes' => 
              array (
                'startLine' => 221,
                'endLine' => 221,
                'startTokenPos' => 409,
                'startFilePos' => 5367,
                'endTokenPos' => 409,
                'endFilePos' => 5370,
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
                      'name' => 'Rubix\\ML\\Kernels\\Distance\\Distance',
                      'isIdentifier' => false,
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
            'startLine' => 221,
            'endLine' => 221,
            'startColumn' => 9,
            'endColumn' => 32,
            'parameterIndex' => 7,
            'isOptional' => true,
          ),
        ),
        'returnsReference' => false,
        'returnType' => NULL,
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * @param int $dimensions
 * @param float $rate
 * @param int $perplexity
 * @param float $exaggeration
 * @param int $epochs
 * @param float $minGradient
 * @param int $window
 * @param Distance|null $kernel
 * @throws InvalidArgumentException
 */',
        'startLine' => 213,
        'endLine' => 272,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Transformers',
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'currentClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'aliasName' => NULL,
      ),
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
 * Return the data types that this transformer is compatible with.
 *
 * @internal
 *
 * @return list<\\Rubix\\ML\\DataType>
 */',
        'startLine' => 281,
        'endLine' => 284,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Transformers',
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'currentClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'aliasName' => NULL,
      ),
      'steps' => 
      array (
        'name' => 'steps',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Generator',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return an iterable progress table with the steps from the last training session.
 *
 * @return Generator<mixed[]>
 */',
        'startLine' => 291,
        'endLine' => 303,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => true,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Transformers',
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'currentClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'aliasName' => NULL,
      ),
      'losses' => 
      array (
        'name' => 'losses',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
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
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the magnitudes of the gradient at each epoch from the last embedding.
 *
 * @return float[]|null
 */',
        'startLine' => 310,
        'endLine' => 313,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Transformers',
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'currentClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'aliasName' => NULL,
      ),
      'transform' => 
      array (
        'name' => 'transform',
        'parameters' => 
        array (
          'samples' => 
          array (
            'name' => 'samples',
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
            'byRef' => true,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 320,
            'endLine' => 320,
            'startColumn' => 31,
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
            'name' => 'void',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Transform the dataset in place.
 *
 * @param array<mixed[]> $samples
 */',
        'startLine' => 320,
        'endLine' => 415,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Transformers',
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'currentClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'aliasName' => NULL,
      ),
      'pairwiseDistances' => 
      array (
        'name' => 'pairwiseDistances',
        'parameters' => 
        array (
          'samples' => 
          array (
            'name' => 'samples',
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
            'startLine' => 423,
            'endLine' => 423,
            'startColumn' => 42,
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
            'name' => 'array',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Calculate the pairwise distances for each sample and return them in a 2-d array.
 *
 * @param array<mixed[]> $samples
 * @return array<float[]>
 */',
        'startLine' => 423,
        'endLine' => 438,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 2,
        'namespace' => 'Rubix\\ML\\Transformers',
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'currentClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'aliasName' => NULL,
      ),
      'affinities' => 
      array (
        'name' => 'affinities',
        'parameters' => 
        array (
          'distances' => 
          array (
            'name' => 'distances',
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
            'startLine' => 447,
            'endLine' => 447,
            'startColumn' => 35,
            'endColumn' => 50,
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
            'name' => 'array',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Compute the conditional probabilities from the distance matrix such that
 * they approximately match the desired perplexity.
 *
 * @param array<float[]> $distances
 * @return array<float[]>
 */',
        'startLine' => 447,
        'endLine' => 513,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 2,
        'namespace' => 'Rubix\\ML\\Transformers',
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'currentClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'aliasName' => NULL,
      ),
      'gradient' => 
      array (
        'name' => 'gradient',
        'parameters' => 
        array (
          'p' => 
          array (
            'name' => 'p',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Tensor\\Matrix',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 523,
            'endLine' => 523,
            'startColumn' => 33,
            'endColumn' => 41,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'y' => 
          array (
            'name' => 'y',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Tensor\\Matrix',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 523,
            'endLine' => 523,
            'startColumn' => 44,
            'endColumn' => 52,
            'parameterIndex' => 1,
            'isOptional' => false,
          ),
          'distances' => 
          array (
            'name' => 'distances',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Tensor\\Matrix',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 523,
            'endLine' => 523,
            'startColumn' => 55,
            'endColumn' => 71,
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
            'name' => 'Tensor\\Matrix',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Compute the gradient of the KL Divergence cost function with respect to the embedding.
 *
 * @param Matrix $p
 * @param Matrix $y
 * @param Matrix $distances
 * @return Matrix
 */',
        'startLine' => 523,
        'endLine' => 543,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 2,
        'namespace' => 'Rubix\\ML\\Transformers',
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'currentClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'aliasName' => NULL,
      ),
      'attenuate' => 
      array (
        'name' => 'attenuate',
        'parameters' => 
        array (
          'gain' => 
          array (
            'name' => 'gain',
            'default' => NULL,
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
            'startLine' => 552,
            'endLine' => 552,
            'startColumn' => 34,
            'endColumn' => 44,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'direction' => 
          array (
            'name' => 'direction',
            'default' => NULL,
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
            'startLine' => 552,
            'endLine' => 552,
            'startColumn' => 47,
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
            'name' => 'float',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Attenuate the momentum signal.
 *
 * @param float $gain
 * @param float $direction
 * @return float
 */',
        'startLine' => 552,
        'endLine' => 559,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 2,
        'namespace' => 'Rubix\\ML\\Transformers',
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'currentClassName' => 'Rubix\\ML\\Transformers\\TSNE',
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
        'startLine' => 568,
        'endLine' => 580,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Transformers',
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TSNE',
        'currentClassName' => 'Rubix\\ML\\Transformers\\TSNE',
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