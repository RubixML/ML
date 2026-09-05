<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;
use Rubix\ML\Exceptions\RuntimeException;
use Rubix\ML\Exceptions\InvalidArgumentException;

use function count;
use function array_values;
use function array_merge;
use function array_fill;
use function array_flip;

/**
 * One Hot Encoder
 *
 * The One Hot Encoder takes a feature column of categorical values and produces an n-d
 * *one-hot* representation where n is equal to the number of unique categories in that
 * column. After the transformation, a 0 in any location indicates that the category
 * represented by that column is not present in the sample whereas a 1 indicates that a
 * category is present. One hot encoding is typically used to convert categorical data to
 * continuous so that it can be used to train a learner that is only compatible with
 * continuous features.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class OneHotEncoder implements Transformer, Stateful, Persistable
{
    use AutotrackRevisions;

    /**
     * The categories that should be ignored.
     *
     * @var list<string|int>
     */
    protected array $ignoredCategories = [
        //
    ];

    /**
     * The set of unique possible categories per feature column of the training set.
     *
     * @var array<int, array<int|string, int>>|null
     */
    protected ?array $categories = null;

    /**
     * Build a new one hot encoder with an array of categories to be ignored during encoding.
     *
     * @param mixed[] $ignoredCategories
     * @throws InvalidArgumentException
     */
    public function __construct(array $ignoredCategories = [])
    {
        foreach ($ignoredCategories as $category) {
            if (!is_string($category) and !is_int($category)) {
                throw new InvalidArgumentException(
                    'Ignored category must be a string or integer, '
                    . gettype($category) . ' found.'
                );
            }
        }

        $this->ignoredCategories = array_values($ignoredCategories);
    }

    /**
     * Return the data types that this transformer is compatible with.
     *
     * @internal
     *
     * @return list<DataType>
     */
    public function compatibility() : array
    {
        return DataType::all();
    }

    /**
     * Is the transformer fitted?
     *
     * @return bool
     */
    public function fitted() : bool
    {
        return isset($this->categories);
    }

    /**
     * Return the categories computed during fitting indexed by feature column.
     *
     * @return array<(int|string)[]>|null
     */
    public function categories() : ?array
    {
        return isset($this->categories) ? array_map('array_flip', $this->categories) : null;
    }

    /**
     * Fit the transformer to a dataset.
     *
     * @param Dataset $dataset
     */
    public function fit(Dataset $dataset) : void
    {
        SamplesAreCompatibleWithTransformer::with($dataset, $this)->check();

        $this->categories = [];

        foreach ($dataset->featureTypes() as $column => $type) {
            if ($type->isCategorical()) {
                $categories = $dataset->feature($column);

                $categories = array_unique($categories);

                if ($this->ignoredCategories) {
                    $categories = array_diff($categories, $this->ignoredCategories);
                }

                $categories = array_values($categories);

                /** @var array<int|string, int> $offsets */
                $offsets = array_flip($categories);

                $this->categories[$column] = $offsets;
            }
        }
    }

    /**
     * Transform the dataset in place.
     *
     * @param list<list<mixed>> $samples
     * @throws RuntimeException
     */
    public function transform(array &$samples) : void
    {
        if ($this->categories === null) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            $vectors = [];

            foreach ($this->categories as $column => $categories) {
                $category = $sample[$column];

                $vector = array_fill(0, count($categories), 0);

                if (isset($categories[$category])) {
                    $vector[$categories[$category]] = 1;
                }

                $vectors[] = $vector;

                unset($sample[$column]);
            }

            $sample = array_merge($sample, ...$vectors);
        }

        unset($sample);
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'One Hot Encoder';
    }
}
