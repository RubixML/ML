<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use InvalidArgumentException;

/**
 * Stop Word Filter
 *
 * Removes user-specified words from any categorical feature columns including blobs
 * of text.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class StopWordFilter implements Transformer
{
    /**
     * A list of stop words to filter out of each text feature.
     *
     * @var string[]
     */
    protected $stopWords;

    /**
     * @param array $stopWords
     * @throws \InvalidArgumentException
     */
    public function __construct(array $stopWords = [])
    {
        $regexs = [];

        foreach ($stopWords as $word) {
            if (!is_string($word)) {
                throw new InvalidArgumentException('Stop word must be a'
                    . ' string, ' . gettype($word) . ' found.');
            }

            $regexs[] = '/\b' . preg_quote($word, '/') . '\b/';
        }

        $this->stopWords = array_combine($stopWords, $regexs) ?: [];
    }

    /**
     * Return the data types that this transformer is compatible with.
     *
     * @return int[]
     */
    public function compatibility() : array
    {
        return DataType::ALL;
    }

    /**
     * Transform the dataset in place.
     *
     * @param array $samples
     */
    public function transform(array &$samples) : void
    {
        foreach ($samples as &$sample) {
            foreach ($sample as &$feature) {
                if (is_string($feature)) {
                    $feature = preg_replace($this->stopWords, '', $feature);
                }
            }
        }
    }
}
