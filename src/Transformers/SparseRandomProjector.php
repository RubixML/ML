<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\Dataset;
use MathPHP\LinearAlgebra\Matrix;
use MathPHP\LinearAlgebra\MatrixFactory;
use InvalidArgumentException;
use RuntimeException;

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

        $n = count(static::DISTRIBUTION) - 1;

        $r = [[]];

        for ($i = 0; $i < $dataset->numColumns(); $i++) {
            for ($j = 0; $j < $this->dimensions; $j++) {
                $r[$i][$j] = static::BETA
                    * static::DISTRIBUTION[random_int(0, $n)];
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
        if (!isset($this->r)) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        $samples = MatrixFactory::create($samples)
            ->multiply($this->r)
            ->getMatrix();
    }
}
