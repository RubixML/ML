<?php

namespace Rubix\ML\Extractors;

use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Traversable;

use function Rubix\ML\iterator_first;
use function filter_var;
use function array_keys;
use function count;
use function explode;
use function fclose;
use function feof;
use function fgets;
use function fopen;
use function is_dir;
use function is_file;
use function is_numeric;
use function is_readable;
use function ltrim;
use function preg_split;
use function rtrim;
use function strlen;
use function strpos;
use function strpbrk;
use function strspn;
use function strcspn;
use function str_contains;
use function str_starts_with;
use function strtolower;
use function strtotime;
use function strtoupper;
use function substr;
use function trim;

use const FILTER_VALIDATE_INT;

/**
 * ARFF
 *
 * The Attribute-Relation File Format (ARFF) is an ASCII text format that is native to the Weka
 * machine learning workbench. Along with being widely used in academic research, ARFF files
 * retain the data type of each column via attribute declarations in the header of the file.
 *
 * > **Note:** Missing values are denoted by an unquoted question mark (?) in the file. Missing
 * > real values are imported as NAN, and missing integer, date, and categorical values are
 * > imported as the categorical placeholder which defaults to '?'.
 *
 * References:
 * [1] I. H. Witten et al. (1999). WEKA - Data Mining with the Java Algorithms for Machine
 * Learning Workbench.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ARFF implements Extractor
{
    /**
     * The symbol used to denote a missing value in the file.
     *
     * @var string
     */
    protected const string MISSING = '?';

    /**
     * The continuous numeric attribute type code.
     *
     * @var int
     */
    protected const int TYPE_FLOAT = 0;

    /**
     * The categorical string attribute type code.
     *
     * @var int
     */
    protected const int TYPE_STRING = 1;

    /**
     * The date attribute type code.
     *
     * @var int
     */
    protected const int TYPE_DATE = 2;

    /**
     * The integer attribute type code.
     *
     * @var int
     */
    protected const int TYPE_INTEGER = 3;

    /**
     * The path to the file on disk.
     *
     * @var non-empty-string
     */
    protected string $path;

    /**
     * The string to substitute in place of missing date and categorical values.
     *
     * @var string|int
     */
    protected string|int $categoricalPlaceholder;

    /**
     * Return the uppercase directive at the beginning of a line.
     *
     * @param string $line
     * @return string
     */
    protected static function directive(string $line) : string
    {
        $parts = preg_split('/\s+/', $line, 2) ?: [$line];

        return strtoupper($parts[0]);
    }

    /**
     * Parse the name and type code of an attribute declaration.
     *
     * @param string $line
     * @param int $lineNumber
     * @throws RuntimeException
     * @return array{string, int}
     */
    protected static function parseAttribute(string $line, int $lineNumber) : array
    {
        [$name, $typespec] = self::token(ltrim(substr($line, strlen('@attribute'))));

        if ($name === '') {
            throw new RuntimeException("Attribute name not found on line $lineNumber.");
        }

        $type = self::attributeType($typespec, $lineNumber);

        return [$name, $type];
    }

    /**
     * Return the attribute type code corresponding to a type specifier.
     *
     * @param string $typespec
     * @param int $line
     * @throws RuntimeException
     * @return int
     */
    protected static function attributeType(string $typespec, int $line) : int
    {
        $typespec = strtolower(trim($typespec));

        if ($typespec === 'numeric' or $typespec === 'real') {
            return self::TYPE_FLOAT;
        }

        if ($typespec === 'integer') {
            return self::TYPE_INTEGER;
        }

        if ($typespec === 'string') {
            return self::TYPE_STRING;
        }

        if ($typespec === 'date' or str_starts_with($typespec, 'date ')) {
            return self::TYPE_DATE;
        }

        if (isset($typespec[0]) and $typespec[0] === '{') {
            return self::TYPE_STRING;
        }

        throw new RuntimeException("Unsupported attribute type '$typespec' on line $line.");
    }

    /**
     * Extract the first token of a string, either a bare token delimited by whitespace or a
     * quoted token. Return the token and the remainder of the string.
     *
     * @param string $string
     * @throws RuntimeException
     * @return array{string, string}
     */
    protected static function token(string $string) : array
    {
        $string = ltrim($string);

        if (isset($string[0]) and self::opensQuote($string, 0)) {
            [$token, $offset] = self::quoted($string, 0);

            return [$token, substr($string, $offset)];
        }

        $parts = preg_split('/\s+/', $string, 2) ?: [$string];

        return [$parts[0], $parts[1] ?? ''];
    }

    /**
     * Read a single or double-quoted string starting at the given offset of the opening quote.
     * A backslash escapes the next character and a doubled quote represents a literal quote.
     * Return the unquoted string and the offset immediately after the closing quote.
     *
     * @param string $string
     * @param int $offset
     * @throws RuntimeException
     * @return array{string, int}
     */
    protected static function quoted(string $string, int $offset) : array
    {
        $quote = $string[$offset];

        $token = '';
        $length = strlen($string);

        for ($i = $offset + 1; $i < $length; ++$i) {
            if ($string[$i] === '\\' and isset($string[$i + 1])) {
                $token .= $string[$i + 1];
                ++$i;

                continue;
            }

            if ($string[$i] !== $quote) {
                $token .= $string[$i];

                continue;
            }

            if (isset($string[$i + 1]) and $string[$i + 1] === $quote) {
                $token .= $quote;
                ++$i;

                continue;
            }

            return [$token, $i + 1];
        }

        throw new RuntimeException('Unterminated quoted string.');
    }

    /**
     * Split a data record into its comma-separated values. Whitespace surrounding a value is
     * ignored and quoted values are unquoted. An unquoted question mark denotes a missing
     * value and is returned as null.
     *
     * @param string $line
     * @param int $lineNumber
     * @throws RuntimeException
     * @return list<string|null>
     */
    protected static function values(string $line, int $lineNumber) : array
    {
        $values = [];

        if (strpbrk($line, '\'"') === false) {
            foreach (explode(',', $line) as $value) {
                $value = trim($value, " \t");

                $values[] = $value === self::MISSING ? null : $value;
            }

            return $values;
        }

        $length = strlen($line);
        $offset = 0;

        while (true) {
            $offset += strspn($line, " \t", $offset);

            if (isset($line[$offset]) and self::opensQuote($line, $offset)) {
                [$value, $offset] = self::quoted($line, $offset);

                $offset += strspn($line, " \t", $offset);

                if ($offset < $length and $line[$offset] !== ',') {
                    throw new RuntimeException("Malformed record on line $lineNumber.");
                }

                $values[] = $value;
            } else {
                $end = strpos($line, ',', $offset);

                if ($end === false) {
                    $end = $length;
                }

                $value = rtrim(substr($line, $offset, $end - $offset), " \t");

                $values[] = $value === self::MISSING ? null : $value;

                $offset = $end;
            }

            if ($offset >= $length) {
                break;
            }

            ++$offset;
        }

        return $values;
    }

    /**
     * Does the character at the given offset open a quoted string? A single or double quote
     * opens a quoted string only at the beginning of a token.
     *
     * @param string $line
     * @param int $offset
     * @return bool
     */
    protected static function opensQuote(string $line, int $offset) : bool
    {
        if ($line[$offset] !== "'" and $line[$offset] !== '"') {
            return false;
        }

        return $offset === 0 or str_contains(" \t\r\n,{", $line[$offset - 1]);
    }

    /**
     * Do the quotes in a line represent a balanced (fully enclosed) string?
     *
     * @param string $line
     * @return bool
     */
    protected static function balanced(string $line) : bool
    {
        $quote = null;
        $length = strlen($line);
        $i = 0;

        while ($i < $length) {
            $i += strcspn($line, $quote === null ? '\\\'"' : '\\' . $quote, $i);

            if ($i >= $length) {
                break;
            }

            if ($line[$i] === '\\') {
                $i += 2;

                continue;
            }

            if ($quote === null) {
                if (self::opensQuote($line, $i)) {
                    $quote = $line[$i];
                }

                ++$i;

                continue;
            }

            if (isset($line[$i + 1]) and $line[$i + 1] === $quote) {
                $i += 2;

                continue;
            }

            $quote = null;

            ++$i;
        }

        return $quote === null;
    }

    /**
     * Strip the comment from a line of text. Comments begin with a percent sign (%) and may
     * appear on their own line or at the end of a record.
     *
     * @param string $line
     * @return string
     */
    protected static function stripComment(string $line) : string
    {
        $quote = null;
        $length = strlen($line);
        $i = 0;

        while ($i < $length) {
            $i += strcspn($line, $quote === null ? '\\\'"%' : '\\' . $quote, $i);

            if ($i >= $length) {
                break;
            }

            if ($line[$i] === '\\') {
                $i += 2;

                continue;
            }

            if ($quote === null) {
                if ($line[$i] === '%') {
                    return substr($line, 0, $i);
                }

                if (self::opensQuote($line, $i)) {
                    $quote = $line[$i];
                }

                ++$i;

                continue;
            }

            if (isset($line[$i + 1]) and $line[$i + 1] === $quote) {
                $i += 2;

                continue;
            }

            $quote = null;

            ++$i;
        }

        return $line;
    }

    /**
     * @param string $path
     * @param string|int $categoricalPlaceholder
     * @throws InvalidArgumentException
     */
    public function __construct(string $path, string|int $categoricalPlaceholder = self::MISSING)
    {
        if (empty($path)) {
            throw new InvalidArgumentException('Path cannot be empty.');
        }

        if (is_dir($path)) {
            throw new InvalidArgumentException('Path must be to a file, folder given.');
        }

        $this->path = $path;
        $this->categoricalPlaceholder = $categoricalPlaceholder;
    }

    /**
     * Return the column titles of the data table.
     *
     * @return array<string>
     */
    public function header() : array
    {
        return array_keys(iterator_first($this));
    }

    /**
     * Return an iterator for the records in the data table.
     *
     * @throws RuntimeException
     * @return \Generator<mixed[]>
     */
    public function getIterator() : Traversable
    {
        if (!is_file($this->path)) {
            throw new RuntimeException("Path {$this->path} is not a file.");
        }

        if (!is_readable($this->path)) {
            throw new RuntimeException("Path {$this->path} is not readable.");
        }

        $handle = fopen($this->path, 'r');

        if (!$handle) {
            throw new RuntimeException('Could not open file pointer.');
        }

        $attributes = [];
        $inHeader = true;
        $line = 0;
        $buffer = '';

        try {
            while (!feof($handle)) {
                $data = fgets($handle);

                if ($data === false) {
                    break;
                }

                ++$line;

                $buffer .= $data;

                if (!self::balanced(self::stripComment($buffer))) {
                    continue;
                }

                $buffer = rtrim($buffer);

                $clean = trim(self::stripComment($buffer));

                $buffer = '';

                if ($clean === '') {
                    continue;
                }

                if ($inHeader) {
                    $directive = self::directive($clean);

                    if ($directive === '@ATTRIBUTE') {
                        [$name, $type] = self::parseAttribute($clean, $line);

                        $attributes[$name] = $type;
                    } elseif ($directive === '@DATA') {
                        $inHeader = false;
                    }
                } else {
                    $values = self::values($clean, $line);

                    if (count($values) !== count($attributes)) {
                        throw new RuntimeException("Malformed record on line $line.");
                    }

                    $record = [];

                    $i = 0;

                    foreach ($attributes as $name => $type) {
                        $value = $values[$i];

                        switch ($type) {
                            case self::TYPE_FLOAT:
                                if ($value !== null) {
                                    if (!is_numeric($value)) {
                                        throw new RuntimeException("Expected numeric value on line $line.");
                                    }

                                    $value = (float) $value;
                                } else {
                                    $value = NAN;
                                }

                                break;

                            case self::TYPE_STRING:
                                if ($value === null) {
                                    $value = $this->categoricalPlaceholder;
                                }

                                break;

                            case self::TYPE_INTEGER:
                                if ($value !== null) {
                                    if (filter_var($value, FILTER_VALIDATE_INT) === false) {
                                        throw new RuntimeException("Expected integer value on line $line.");
                                    }

                                    $value = (int) $value;
                                } else {
                                    $value = $this->categoricalPlaceholder;
                                }

                                break;

                            case self::TYPE_DATE:
                                if ($value !== null) {
                                    $timestamp = strtotime($value);

                                    if ($timestamp === false) {
                                        throw new RuntimeException("Invalid date on line $line.");
                                    }

                                    $value = (float) $timestamp;
                                } else {
                                    $value = $this->categoricalPlaceholder;
                                }

                                break;
                        }

                        $record[$name] = $value;

                        ++$i;
                    }

                    yield $record;
                }
            }

            if ($buffer !== '') {
                throw new RuntimeException("Unterminated quoted string on line $line.");
            }
        } finally {
            fclose($handle);
        }
    }
}
