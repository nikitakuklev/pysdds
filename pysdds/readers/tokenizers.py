def tokenize_namelist(line):
    assert line[0] == '&', f'Unexpected namelist start {line[0]}'
    tags = []
    keys = []
    values = []
    buffer = ''
    i, j = 0, 0
    # TODO: state machine enums
    mode_tag = True
    saw_tag_start = False
    mode_kv = False
    kv_key_section = True
    mode_gap = False
    parsing_tag_or_kv = False
    value_quote_mode = False
    value_escape_next = False
    value_parse_started = False

    while True:
        if mode_gap:
            #print('gap', j)
            if j >= len(line):
                #print('end')
                break
            if line[j] != ' ':
                i = j
                mode_gap = False
            else:
                j += 1
        elif not parsing_tag_or_kv:
            if line[j] == '&':
                #print('call tag')
                mode_tag = True
            else:
                if len(tags) == 0:
                    raise Exception("Expect first element to be a tag")
                #print('call kv')
                mode_kv = True
            parsing_tag_or_kv = True
        elif mode_tag:
            #print('tag',j)
            if not saw_tag_start:
                assert line[j] == '&'
                saw_tag_start = True
                j += 1
            if j >= len(line) or line[j] == ' ':
                tags.append(line[i:j])
                i = j
                mode_gap = True
                mode_tag = False
                parsing_tag_or_kv = False
                saw_tag_start = False
            else:
                j += 1
        elif mode_kv:
            if kv_key_section:
                #print('k', j)
                if line[j] == '=':
                    kv_key_section = False
                    keys.append(line[i:j])
                    j += 1
                    i = j
                else:
                    j += 1
            else:
                #print('v', j)
                currentchar = line[j]

                # parse c-style string expression
                if not value_parse_started:
                    value_parse_started = True
                    if currentchar == '"':
                        value_quote_mode = True
                        #print('vquoteon', j)
                        j += 1
                        i = j
                    continue
                if value_quote_mode:
                    nextchar = line[j + 1]
                    if value_escape_next:
                        buffer += currentchar
                        j += 1
                        value_escape_next = False
                    else:
                        if currentchar == '\\':
                            # escape next char
                            value_escape_next = True
                            j += 1
                        elif currentchar == '"':
                            # end of quoted segment
                            if nextchar != ',':
                                raise Exception(f'Missing delimiter "," at end of buffer {buffer}')
                            values.append(buffer)
                            buffer = ''
                            j += 2
                            value_quote_mode = False
                            value_parse_started = False
                            mode_kv = False
                            mode_gap = True
                            kv_key_section = True
                            parsing_tag_or_kv = False
                        else:
                            # accept char
                            buffer += currentchar
                            j += 1
                else:
                    nextchar = line[j + 1]
                    if currentchar == '\\':
                        if nextchar == '"' or nextchar == '\\':
                            buffer += nextchar
                            j += 2
                        else:
                            buffer += currentchar
                            j += 1
                    elif currentchar == ',':
                        value_parse_started = False
                        mode_kv = False
                        mode_gap = True
                        kv_key_section = True
                        parsing_tag_or_kv = False
                        values.append(buffer)
                        buffer = ''
                        j += 1
                        i = j
                    else:
                        buffer += currentchar
                        j += 1
        else:
            raise Exception('State machine internal error')

    return tags, keys, values
