from pysdds.util.errors import SDDSReadError

delimiters = {"\r", "\n", " "}
delimiterscomma = {",", "\r", "\n", " "}

def tokenize_namelist(line):
    assert line[0] == "&", f"Unexpected namelist start {line[0]}"
    tags = []
    keys = []
    values = []
    buffer = ""
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
    value_parse_space_seen = False

    while True:
        if mode_gap:
            # gap between tag/key/value, i.e. space
            # print('gap', j)
            if j >= len(line):
                # print('end')
                break
            if line[j] not in delimiters:
                i = j
                mode_gap = False
            else:
                j += 1
        elif not parsing_tag_or_kv:
            if line[j] == "&":
                # print('call tag')
                mode_tag = True
            else:
                if len(tags) == 0:
                    raise SDDSReadError("Expect first element to be a tag")
                # print('call kv')
                mode_kv = True
            parsing_tag_or_kv = True
        elif mode_tag:
            # print('tag',j)
            if not saw_tag_start:
                assert line[j] == "&", f"Unexpected tag start {line[j]=} at {j}"
                saw_tag_start = True
                j += 1
            if j >= len(line) or line[j] in delimiters:
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
                # print('k', j)
                if line[j] == "=":
                    kv_key_section = False
                    keys.append(line[i:j])
                    j += 1
                    i = j
                else:
                    j += 1
            else:
                # print('v', j)
                currentchar = line[j]

                # parse c-style string expression
                if not value_parse_started:
                    value_parse_started = True
                    if currentchar == '"':
                        value_quote_mode = True
                        # print('vquoteon', j)
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
                        if currentchar == "\\":
                            # escape next char
                            value_escape_next = True
                            j += 1
                        elif currentchar == '"':
                            # end of quoted segment and value
                            values.append(buffer)
                            buffer = ""
                            j += 2
                            i = j
                            value_parse_started = mode_kv = value_quote_mode = False
                            kv_key_section = True
                            #print("end of quoted segment", values[-1])

                            if nextchar != ",":
                                # might not have comma if only a single item or at the end
                                # so scan ahead to find next state
                                found_next_item_or_end = False
                                for jmp in range(0, 20):
                                    c = line[j + jmp]
                                    #print("nextscan ", jmp, c)
                                    if c in delimiters:
                                        continue
                                    elif c == ",":
                                        mode_gap = found_next_item_or_end = True
                                        parsing_tag_or_kv = False
                                        j += jmp
                                        i = j
                                        break
                                    elif c == "&":
                                        # found &end tag hopefully
                                        mode_tag = parsing_tag_or_kv = found_next_item_or_end = True
                                        j += jmp
                                        i = j
                                        break
                                    else:
                                        raise ValueError(
                                            f"Invalid char [{c}] after quoted string"
                                        )
                                if not found_next_item_or_end:
                                    raise ValueError(
                                        "Too many spaces after quoted string?"
                                    )
                            else:
                                parsing_tag_or_kv = False
                                mode_gap = True
                        else:
                            # accept char
                            buffer += currentchar
                            j += 1
                else:
                    nextchar = line[j + 1]
                    # if currentchar == ' ':
                    #     value_parse_space_seen = True
                    #     j += 1
                    #     continue
                    if currentchar == "\\":
                        if nextchar == '"' or nextchar == "\\":
                            buffer += nextchar
                            j += 2
                        else:
                            buffer += currentchar
                            j += 1
                    elif currentchar in delimiterscomma:
                        value_parse_started = mode_kv = parsing_tag_or_kv = False
                        mode_gap = kv_key_section = True
                        values.append(buffer)
                        buffer = ""
                        j += 1
                        i = j
                    elif currentchar == "&":
                        value_parse_started = mode_kv = mode_gap = False
                        mode_tag = parsing_tag_or_kv = True
                        values.append(buffer)
                        buffer = ""
                        i = j
                    else:
                        # if value_parse_space_seen:
                        #    #assume that actually next namelist starts here after space separator
                        #
                        #    raise ValueError(f'Got character after unquoted space')
                        buffer += currentchar
                        j += 1
        else:
            raise ValueError("State machine internal error")

    return tags, keys, values
