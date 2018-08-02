<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\Dataset;
use MathPHP\LinearAlgebra\Matrix;
use MathPHP\LinearAlgebra\MatrixFactory;
use InvalidArgumentException;
use RuntimeException;

/**
 * Sparse Random Projector
 *
 * A Random Projector is a dimensionality reducer based on the
 * Johnson-Lindenstrauss lemma that uses a random matrix to project a feature
 * vector onto a user-specified number of dimensions. It is faster than most
 * non-randomized dimensionality reduction techniques and offers similar
 * performance. The sparse version uses a random matrix sampled from a sparse
 * uniform distribution (mostly 0s).
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class SparseRandomProjector implements Transformer
{
    const DISTRIBUTION = [-self::ROOT_3, 0.0, 0.0, 0.0, 0.0, self::ROOT_3];

    const ROOT_3 = 1.73205080757;

    /**
     * The target number of dimensions.
     *
     * @var int
     */
    protected $dimensions;

    /**
     * The randomized matrix R.
     *
     * @var \MathPHP\LinearAlgebra\Matrix|null
    */
    protected $r;

    /**
     * Calculate the minimum number of dimensions for n total samples with a
     * given maximum distortion using the Johnson-Lindenstrauss lemma.
     *
     * @param  int  $n
     * @param  float  $maxDistortion
     * @return int
     */
    public static function minDimensions(int $n, float $maxDistortion = 0.1) : int
    {
        return (int) round(4.0 * log($n)
            / ($maxDistortion ** 2 / 2 - $maxDistortion ** 3 / 3));
    }

    /**
     * @param  int  $dimensions
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $dimensions)
    {
        if ($dimensions < 1) {
            throw new InvalidArgumentException('Cannot project onto less than'
                . ' 1 dimension.');
        }

        $this->dimensions = $dimensions;
    }

    /**
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function fit(Dataset $dataset) : void
    {
        if (in_array(self::CATEGORICAL, $dataset->columnTypes())) {
            throw new InvalidArgumentException('This transformer only works'
                . ' with continuous features.');
        }

        $columns = $dataset->numColumns();

        $n = count(static::DISTRIBUTION) - 1;

        $r = [[]];

        for ($i = 0; $i < $columns; $i++) {
            for ($j = 0; $j < $this->dimensions; $j++) {
                $r[$i][$j] = static::DISTRIBUTION[rand(0, $n)];
            }
        }

        $this->r = new Matrix($r);
    }

    /**
     * Transform each sample into a dense polynomial feature vector in the degree
     * given.
     *
     * @param  array  $samples
     * @throws \RuntimeException
     * @return void
     */
    public function transform(array &$samples) : void
    {
        if (is_null($this->r)) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        $samples = MatrixFactory::create($samples)
            ->multiply($this->r)
            ->getMatrix();
    }
}
