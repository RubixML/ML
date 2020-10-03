<?php

namespace Rubix\ML\Extractors;

use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Rubix\ML\Storage\Exceptions\StorageException;
use Rubix\ML\Storage\LocalFilesystem;
use Rubix\ML\Storage\Reader;
use Rubix\ML\Storage\ReadProxy;
use Rubix\ML\Storage\Streams\Stream;
use Generator;

use function strlen;

/**
 * CSV
 *
 * A plain-text format that use newlines to delineate rows and a user-specified delimiter
 * (usually a comma) to separate the values of each column in a data table. Comma-Separated
 * Values (CSV) format is a common format but suffers from not being able to retain type
 * information - thus, all data is imported as categorical data (strings) by default.
 *
 * > **Note:** This implementation of CSV is based on the definition in RFC 4180.
 *
 * References:
 * [1] Y. Shafranovich. (2005). Common Format and MIME Type for Comma-Separated Values (CSV)
 * Files.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class CSV implements Extractor
{
    /**
     * A readable stream object pointing to the underlying CSV data
     *
     * @var \Rubix\ML\Storage\Streams\Stream
     */
    protected $stream;

    /**
     * Does the CSV document have a header as the first row?
     *
     * @var bool
     */
    protected $header;

    /**
     * The character that delineates the values of the columns of the data table.
     *
     * @var string
     */
    protected $delimiter;

    /**
     * The character used to enclose a cell that contains a delimiter in the body.
     *
     * @var string
     */
    protected $enclosure;

    /**
     * @param string $location
     * @param bool $header
     * @param string $delimiter
     * @param string $enclosure
     * @param ?Reader $storage
     * @throws \InvalidArgumentException
     * @throws \Rubix\ML\Storage\Exceptions\StorageException
     */
    public function __construct(
        string $location,
        bool $header = false,
        string $delimiter = ',',
        string $enclosure = '"',
        ?Reader $storage = null
    ) {
        if (!$storage) {
            $storage = new LocalFilesystem();
        }

        $storage = new ReadProxy($storage);

        if (!$storage->exists($location)) {
            throw new InvalidArgumentException("Location $location does not exist.");
        }

        if (strlen($delimiter) !== 1) {
            throw new InvalidArgumentException('Delimiter must be'
                . ' a single character, ' . strlen($delimiter) . ' given.');
        }

        if (strlen($enclosure) !== 1) {
            throw new InvalidArgumentException('Enclosure must be'
                . ' a single character, ' . strlen($enclosure) . ' given.');
        }

        $this->stream = $storage->read($location, Stream::READ_ONLY);
        $this->header = $header;
        $this->delimiter = $delimiter;
        $this->enclosure = $enclosure;
    }

    /**
     * Clean up the any open file handles.
     */
    public function __destruct()
    {
        try {
            if ($this->stream and $this->stream->open()) {
                $this->stream->close();
            }
        } catch (StorageException $e) {
            //
        }
    }

    /**
     * Return an iterator for the records in the data table.
     *
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @throws \Rubix\ML\Storage\Exceptions\StorageException
     * @return \Generator<mixed[]>
     */
    public function getIterator() : Generator
    {
        if ($this->stream->seekable()) {
            $this->stream->rewind();
        }

        $line = 0;

        if ($this->header) {
            $row = $this->stream->line();

            if (empty(trim($row))) {
                throw new RuntimeException('Header not found on the first line.');
            }

            $header = str_getcsv($row, $this->delimiter, $this->enclosure);
            ++$line;
        }

        foreach ($this->stream as $row) {
            if (empty($row)) {
                continue 1;
            }

            $record = str_getcsv($row, $this->delimiter, $this->enclosure);
            ++$line;

            if (isset($header)) {
                $record = array_combine($header, $record);
            }

            if (!$record) {
                throw new RuntimeException("Malformed record on line $line.");
            }

            yield $record;
        }
    }
}
