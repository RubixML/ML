<?php

namespace Rubix\Engine\Transformers;

use Rubix\Engine\Datasets\Dataset;
use InvalidArgumentException;

class DensePolynomialExpander implements Transformer
{
    /**
     * The degree of the polynomials to generate. Higher order polynomials are
     * able to fit nonlinear data better, however require extra features to be added
     * to the dataset.
     *
     * @var int
     */
    protected $degree;

    /**
     * Should we add a bias term to the feature vector?
     *
     * @var bool
     */
    protected $bias;

    /**
     * The number of columns in the fitted dataset.
     *
     * @var int
     */
    protected $columns;

    /**
     * @param  int  $degree
     * @return void
     */
    public function __construct(int $degree = 2, bool $bias = true)
    {
        $this->degree = $degree;
        $this->bias = $bias;
    }

    /**
     * @param  \Rubix\Engine\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function fit(Dataset $dataset) : void
    {
        if (in_array(self::CATEGORICAL, $dataset->columnTypes())) {
            throw new InvalidArgumentException('This transformer only works'
                . ' with continuous features.');
        }

        $this->columns = $dataset->numColumns();
    }

    /**
     * Transform each sample into a dense polynomial feature vector in the degree
     * given.
     *
     * @param  array  $samples
     * @return void
     */
    public function transform(array &$samples) : void
    {
        foreach ($samples as &$sample) {
            $vector = [];

            if ($this->bias) {
                $vector[] = 1.0;
            }

            for ($i = 0; $i < $this->columns; $i++) {
                for ($j = $this->degree; $j > 0; $j--) {
                    $vector[] = pow($sample[$i], $j);
                }
            }

            $sample = $vector;
        }
    }
}
