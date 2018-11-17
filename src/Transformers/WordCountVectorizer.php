<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\DataFrame;
use Rubix\ML\Other\Tokenizers\Word;
use Rubix\ML\Other\Tokenizers\Tokenizer;
use InvalidArgumentException;
use RuntimeException;

/**
 * Word Count Vectorizer
 *
 * In machine learning, word counts are often used to represent natural language
 * as numerical vectors. The Word Count Vectorizer builds a vocabulary using
 * hash tables from the training samples during fitting and transforms an array
 * of strings (text blobs) into sparse feature vectors. Each feature column
 * represents a word from the vocabulary and the value denotes the number of times
 * that word appears in a given sample.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class WordCountVectorizer implements Stateful
{
    /**
     * The maximum size of the vocabulary.
     *
     * @var int
     */
    protected $maxVocabulary;

    /**
     * The minimum number of documents a word must appear in to be added to
     * the vocabulary.
     * 
     * @var int
     */
    protected $minDocumentFrequency;

    /**
     * An array of stop words i.e. words to filter out of the original text.
     *
     * @var array
     */
    protected $stopWords = [
        //
    ];

    /**
     * Should the text be normalized before tokenized? i.e. remove extra
     * whitespace and lowercase.
     *
     * @var bool
     */
    protected $normalize;

    /**
     * The tokenizer used to extract text data into tokenable values.
     *
     * @var \Rubix\ML\Other\Tokenizers\Tokenizer
     */
    protected $tokenizer;

    /**
     * The vocabulary of the fitted training set per categorical  feature
     * column.
     *
     * @var array|null
     */
    protected $vocabulary;

    /**
     * @param  int  $maxVocabulary
     * @param  int  $minDocumentFrequency
     * @param  array  $stopWords
     * @param  bool  $normalize
     * @param  \Rubix\ML\Other\Tokenizers\Tokenizer|null  $tokenizer
     * @return void
     */
    public function __construct(int $maxVocabulary = PHP_INT_MAX, int $minDocumentFrequency = 1, array $stopWords = [],
                                bool $normalize = true, ?Tokenizer $tokenizer = null)
    {
        if ($maxVocabulary < 1) {
            throw new InvalidArgumentException('The size of the vocabular must'
                . " be at least 1 word, $maxVocabulary given.");
        }

        if ($minDocumentFrequency < 1) {
            throw new InvalidArgumentException('Minimum document frequency must'
            . " be at least 1, $minDocumentFrequency given.");
        }

        foreach ($stopWords as $word) {
            if (!is_string($word)) {
                throw new InvalidArgumentException('Stop word must be a string, '
                    . gettype($word) . ' found.');
            }
        }

        if ($normalize) {
            $stopWords = array_map('trim', array_map('strtolower', $stopWords));
        }

        if (is_null($tokenizer)) {
            $tokenizer = new Word();
        }

        $this->maxVocabulary = $maxVocabulary;
        $this->minDocumentFrequency = $minDocumentFrequency;
        $this->stopWords = array_flip(array_unique($stopWords));
        $this->normalize = $normalize;
        $this->tokenizer = $tokenizer;
    }

    /**
     * Return an array of words in the vocabulary.
     *
     * @return array
     */
    public function vocabulary() : array
    {
        return array_flip($this->vocabulary ?? []);
    }

    /**
     * Return the size of the vocabulary in words.
     *
     * @return int
     */
    public function size() : int
    {
        return count($this->vocabulary ?? []);
    }

    /**
     * Fit the transformer to the dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return void
     */
    public function fit(Dataset $dataset) : void
    {
        $columns = $dataset->columnsByType(DataFrame::CATEGORICAL);

        $this->vocabulary = [];

        foreach ($columns as $column => $values) {
            $tfs = $dfs = [];

            foreach ($values as $text) {
                if ($this->normalize) {
                    $text = preg_replace('/\s+/', ' ', strtolower($text));
                }

                $tokens = $this->tokenizer->tokenize($text);

                $counts = array_count_values($tokens);

                foreach ($counts as $token => $count) {
                    if (!isset($this->stopWords[$token])) {
                        if (isset($tfs[$token])) {
                            $tfs[$token] += $count;
                            $dfs[$token] += 1;
                        } else {
                            $tfs[$token] = $count;
                            $dfs[$token] = 1;
                        }
                    }
                }
            }

            if ($this->minDocumentFrequency > 1) {
                foreach ($dfs as $token => $frequency) {
                    if ($frequency < $this->minDocumentFrequency) {
                        unset($tfs[$token]);
                    }
                }
            }

            if (count($tfs) > $this->maxVocabulary) {
                arsort($tfs);
    
                $tfs = array_slice($tfs, 0, $this->maxVocabulary, true);
            }
    
            $this->vocabulary[$column] = array_combine(array_keys($tfs),
                range(0, count($tfs) - 1)) ?: [];
        }
    }

    /**
     * Transform the dataset in place.
     *
     * @param  array  $samples
     * @param  array|null  $labels
     * @throws \RuntimeException
     * @return void
     */
    public function transform(array &$samples, ?array &$labels = null) : void
    {
        if (is_null($this->vocabulary)) {
            throw new RuntimeException('Transformer is not fitted.');
        }

        foreach ($samples as &$sample) {
            $vectors = [];

            foreach ($this->vocabulary as $column => $vocabulary) {
                $text = $sample[$column];

                $vector = array_fill(0, count($vocabulary), 0);

                if ($this->normalize) {
                    $text = preg_replace('/\s+/', ' ', strtolower($text)) ?? '';
                }

                $tokens = $this->tokenizer->tokenize($text);

                $counts = array_count_values($tokens);

                foreach ($counts as $token => $count) {
                    if (isset($vocabulary[$token])) {
                        $vector[$vocabulary[$token]] += $count;
                    }
                }

                $vectors[] = $vector;

                unset($sample[$column]);
            }

            $sample = array_merge($sample, ...$vectors);
        }
    }
}
