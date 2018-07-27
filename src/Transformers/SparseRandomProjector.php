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
 * performance. The difference between the Dense and Sparse Random Projectors
 * are that the Dense version uses a dense random guassian distribution and the
 * Sparse version uses a sparse matrix (mostly 0’s).
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class SparseRandomProjector implements Transformer
{
    const BETA = 1.73205080757;

    const DISTRIBUTION = [-1, 0, 0, 0, 0, 1];

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
                $r[$i][$j] = static::BETA * static::DISTRIBUTION[rand(0, $n)];
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
