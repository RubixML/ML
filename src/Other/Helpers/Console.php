<?php

namespace Rubix\ML\Other\Helpers;

class Console
{
    protected const SIZE_ROWS_COMMAND = 'tput lines';
    protected const SIZE_COLUMNS_COMMAND = 'tput cols';

    protected const DEFAULT_SIZE_ROWS = 24;
    protected const DEFAULT_SIZE_COLUMNS = 80;

    public const TABLE_CELL_WIDTH = 11;
    
    protected const TABLE_CELL_PREFIX = '| ';
    protected const TABLE_CELL_SUFFIX = ' ';

    /**
     * Return the size of the console window in a 2-tuple.
     *
     * @return int[]
     */
    public static function size() : array
    {
        return [
            (int) exec(self::SIZE_ROWS_COMMAND) ?: self::DEFAULT_SIZE_ROWS,
            (int) exec(self::SIZE_COLUMNS_COMMAND) ?: self::DEFAULT_SIZE_COLUMNS,
        ];
    }

    /**
     * Return the string representation of a 2-dimensional data table.
     *
     * @param array[] $table
     * @return string
     */
    public static function table(array $table) : string
    {
        return array_reduce($table, [self::class, 'implodeRow'], '') . PHP_EOL;
    }

    /**
     * Implode a row of the table and return the output.
     *
     * @param string $carry
     * @param mixed[] $row
     * @return string
     */
    protected static function implodeRow(string $carry, array $row) : string
    {
        $border = str_repeat(str_repeat('-', self::TABLE_CELL_WIDTH + 3), count($row)) . '-';

        $temp = array_reduce(array_map('strval', $row), [self::class, 'formatCell'], '') . '|';

        return $carry . PHP_EOL . $temp . PHP_EOL . $border;
    }

    /**
     * Format a cell of a table.
     *
     * @param string $carry
     * @param string $value
     * @return string
     */
    protected static function formatCell(string $carry, string $value) : string
    {
        $value = str_pad(substr($value, 0, self::TABLE_CELL_WIDTH), self::TABLE_CELL_WIDTH);
    
        return $carry . self::TABLE_CELL_PREFIX . $value . self::TABLE_CELL_SUFFIX;
    }
}
